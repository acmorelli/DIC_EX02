import json
import re
from pyspark import SparkContext

# # # # # # # # # # # # #
# input files
#input_file = "hdfs:///user/dic25_shared/amazon-reviews/full/reviews_devset.json"
input_file = "/Users/annaclara/Documents/TUW/SS25/DIC/EX02/input_files/reviews_devset.json"
stop_words_file = "/Users/annaclara/Documents/TUW/SS25/DIC/EX02/input_files/stopwords.txt"

# load the data
rdd_raw = sc.textFile(input_file)
stopwords = set(sc.textFile(stop_words_file).collect())

# filter rdd - only reviews with category and reviewText
rdd= rdd_raw.map(lambda line: json.loads(line)) \
    .filter(lambda d: "category" in d and "reviewText" in d)

def tokenize(text, stopwords):
    tokens = re.split(r"[\s\d(){}\[\].!?,;:+=\-_'\"`~#@&*%€$§\\/]+", text.lower())
    return [t for t in tokens if t and t not in stopwords and len(t) > 1]
# # # # # # # # # # # # #
"""
    word_category_counts: ((word, category), count)

    01.map returns dicts with keys "category" and "reviewText"
    
    02.filter removes incomplete or empty reviews
    
    03.flatMap tokenizes the reviewText,creates a tuple (word, category), creates a list of kv pairs [((word, category), 1), ...]
    --> here one input review creates multiple outputs (one for each token (aka word)
    --> those outputs are flatened into one list
    --> similar to mapper in Hadoop, but here we have a list of tuples
    --> it's a local computation - no shuffling across the cluster partitions
    
    04.reduceByKey sums the counts for each (word, category) pair and groups all kv pairs with the same key
    --> accepts only a flat stream of tuples
    --> like the reducer in Hadoop, but:
    --> the output is a list of tuples ((word, category), count)

    
"""
word_category_counts = rdd.map(lambda line: json.loads(line)) \
    .flatMap(lambda d: [((word, d["category"]), 1) for word in tokenize(d["reviewText"], stopwords)]) \
    .reduceByKey(lambda a, b: a + b)

"""
doc_counts: RDD[Tuple[str, int]] -> number of documents in each category, (category, count)  per line
--> .map: dict with keys "category" and "reviewText" for each review
--> .map: list of tuples (category, count) -- emits one pair of (category, 1) for each review
--> .reduceByKey: sums the counts of documents for each category
"""
doc_counts = rdd.map(lambda line: json.loads(line)) \
    .map(lambda d: (d["category"], 1)) \
    .reduceByKey(lambda a, b: a + b)

total_docs = doc_counts.map(lambda x: x[1]).sum() # list(int) - access count from (cat, count) pairs and sum them up
"""collect() gets data from all partitions and returns it as a list to the driver
possible problem if we have many categories in the data

dict() converts the list of tuples (cat, count) into a dict 
REASON: dicts are faster than lists to be looked up
"""
category_doc_counts = dict(doc_counts.collect()) # transform the rdd into a dict with category as key and count as value


# word_category_counts: ((word, category), count)
# → (word, (category, count))
"""
Returns
word_grouped: RDD (word, [(category, count), ...])

----
--> word_category_counts: ((word, category), count)
--> lambda x: (x[0][0], (x[0][1], x[1])):
    x[0][0] = word (key) -- access the first element of the tuple (word, category) and return the word
    x[0][1] = category
    x[1] = count

01. word_category_counts.map(lambda x) returns a tuple (word, (category, count))
02. groupByKey() groups the tuples by key (word) and returns an RDD (word, [(category, count), ...])
"""
word_grouped = word_category_counts.map(lambda x: (x[0][0], (x[0][1], x[1]))) \
    .groupByKey()

# # # # # # # # # # # # #
def compute_chi2(word_data):
    """
    Params
    word_data: (word, [(category, count), ...]) # processed in all partitions separately
    --- dummy: "book", [("Books", 15), ("Electronics", 3)])

    -------
    Returns
        results: list of tuples ((category, word), chi2_score)

    --> in one call:
        [ (("Books", "book"), 4.28), (("Electronics", "book"), 2.61)), ...]
    --> But flatMap flattens from all partitions, so we get:
    [
    (("Books", "book"), 4.28),
    (("Electronics", "book"), 2.61),
    (("Electronics", "charger"), 6.14),
    ...
        ]
    """
    word, category_counts = word_data
    category_counts = list(category_counts)
    total_word_count = sum([c for _, c in category_counts])

    results = []
    for category, A in category_counts:
        B = total_word_count - A
        C = category_doc_counts[category] - A
        D = (total_docs - category_doc_counts[category]) - B

        numerator = (A * D - B * C) ** 2
        denominator = (A + B) * (C + D) * (A + C) * (B + D)
        if denominator != 0:
            chi = total_docs * numerator / denominator
            results.append(((category, word), chi))
        else:
            print("Something went wrong: denominator is 0")
    return results

"""
word_grouped: RDD (word, [(category, count), ...])

So, word_grouped.flatMap in spark is doing under the hood:
for record in word_grouped:
    compute_chi2(record)

being record each tuple of (word, [(category, count), ...])

In other words, flatMap just calls compute_chi2() for each record of word_grouped

The output is a RDD of tuples ((category, word), chi2_score)
"""
chi_scores = word_grouped.flatMap(compute_chi2)


"""
01. map chi_scores
    -->chi_scores were like ((category, word), chi2_score)
    So we have:
    --> chi_scores.map(lambda x: (x[0][0], (x[1], x[0][1]))):
        x[0][0] = category
        x[0][1] = word
        x[1] = chi2_score
    We get:
    chi_scores.map(category, (chi2_score, word)) --> produces a new reformatted RDD 
    REASON: we need to reformat chi_scores to input it in groupByKey() 

02. groupByKey(): group the tuples by category
    -->Params: RDD (category, (chi2_score, word))
    -->Returns: (category, [(chi2_score, word1), (chi2_score, word2)..])

Examples are easier to understand:
input: [("Books", (4.28, "book")), ("Books", (3.41, "pen"))...]
groupByKey() returns (key, iter object), so we get:

output:
    [ ("Books", [(4.28, "book"), (3.41, "pen"), (2.71, "notebook")]),
      ...]

03. mapValues():
    --> Returns: sorted by chi score (category, [(chi2_score, word1), (chi2_score, word2)..])
    --> doesnt change the key, and a function only to values in the iter object
    --> so it can sort the words inside a category by chi score 
    --> and we dont have to unpack the tuple. It is doing:
    rdd.map(lambda kv: (kv[0], f(kv[1])))
    -
"""
top_words = chi_scores.map(lambda x: (x[0][0], (x[1], x[0][1]))) \
.groupByKey() \
.mapValues(lambda scores: sorted(scores, reverse=True)[:75])

top_words.saveAsTextFile("output_rdd.txt")



