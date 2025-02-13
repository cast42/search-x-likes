# 📊 Comparison of Recall@K, NDCG@K, and MRR

Each metric measures **how well a ranking system performs**, but they focus on **different aspects** of relevance. Let’s compare them using our **toy box analogy**:

| **Metric**   | **What it Measures**                                                     | **Example Focus**                                                                           | **Best for…**                                    |
| ------------ | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| **Recall@K** | ✅ _Did I find my favorite toys at all?_ (**ignores order**)             | Out of 10 favorite toys, how many did I find in my **K picks**?                             | **Coverage of relevant items**                   |
| **NDCG@K**   | 🔝 _How well were my favorite toys ranked?_ (**higher-ranked = better**) | If my absolute favorite toy is ranked **higher**, I get a better score                      | **Ranking quality** (more weight to top results) |
| **MRR**      | 🎯 _How quickly did I find my first favorite toy?_                       | If I find my favorite toy in the **1st pick**, it’s perfect; if in the **5th**, not so good | **First-hit success**                            |

---

## 🔍 **Key Differences**

- **Recall@K** → Only cares about **how many** relevant items are in the top-K but **ignores order**.
- **NDCG@K** → Considers both **relevance** and **order**, rewarding results where the most important items appear **higher**.
- **MRR** → Looks at **how soon** the first relevant result appears—**only the first hit matters**.

---

## 🎯 **When to Use Each Metric?**

- **Use Recall@K** if you care about finding **as many relevant items as possible**, regardless of order (_e.g., product recommendations_).
- **Use NDCG@K** if you care about ranking **the most important items at the top** (_e.g., search engine results_).
- **Use MRR** if finding **a single highly relevant item quickly** is most important (_e.g., question-answering systems_).

Would you like an example using real data to illustrate this? 🚀
