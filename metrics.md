# 📊 Comparison of Precision@K, Recall@K, NDCG@K, and MRR

Each metric measures **how well a ranking system performs**, but they focus on **different aspects** of relevance. Let’s compare them using our **toy box analogy**:

| **Metric**         | **What it Measures**                                                     | **Example Focus**                                                                           | **Best for…**                                    |
| ------------------ | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| **Precision@K** 🎯 | ✅ _Out of the K toys I picked, how many were actually my favorites?_    | If I pick **5 toys** and **3 are favorites**, Precision@5 = **3/5 = 60%**                   | **How accurate the top-K results are**           |
| **Recall@K** 🔍    | ✅ _Did I find my favorite toys at all?_ (**ignores order**)             | Out of **10 favorite toys**, how many did I find in my **K picks**?                         | **Coverage of relevant items**                   |
| **NDCG@K** 🔝      | 🔝 _How well were my favorite toys ranked?_ (**higher-ranked = better**) | If my absolute favorite toy is ranked **higher**, I get a better score                      | **Ranking quality** (more weight to top results) |
| **MRR** 🚀         | 🎯 _How quickly did I find my first favorite toy?_                       | If I find my favorite toy in the **1st pick**, it’s perfect; if in the **5th**, not so good | **First-hit success**                            |

---

## 🔍 **Key Differences**

- **Precision@K** → Measures **how many of the selected K items are actually relevant**, but ignores whether you missed some.
- **Recall@K** → Measures **how many relevant items you found**, but ignores how many irrelevant ones were picked.
- **NDCG@K** → Considers both **relevance** and **order**, rewarding results where the most important items appear **higher**.
- **MRR** → Looks at **how soon** the first relevant result appears—**only the first hit matters**.

---

## 🎯 **When to Use Each Metric?**

- **Use Precision@K** if you care about **how accurate** the top-K results are (_e.g., minimizing false positives in recommendations_).
- **Use Recall@K** if you care about **finding as many relevant items as possible**, regardless of order (_e.g., product recommendations_).
- **Use NDCG@K** if you care about **ranking the most important items at the top** (_e.g., search engine results_).
- **Use MRR** if finding **a single highly relevant item quickly** is most important (_e.g., question-answering systems_).

Would you like an example using real data to illustrate this? 🚀
