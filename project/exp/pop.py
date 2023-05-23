# Calculate Popularity
popularity = df.groupby("item_id").size().reset_index(name="popularity")
popularity.sort_values("popularity", ascending=False, inplace=True)
pop = popularity.item_id.tolist()[:100]

# %%
pop_sum_ndcg_5, pop_sum_ndcg_10, pop_sum_ndcg_20 = 0, 0, 0
pop_sum_recall_5, pop_sum_recall_10, pop_sum_recall_20 = 0, 0, 0
for user in tqdm(dataset.test_seqs):
    # bought_item = df[df.user_id == user].item_id.unique().tolist()
    bought_item = []
    pop_not_bought = [item for item in pop if item not in bought_item]

    true_list = dataset.test_seqs[user]

    pop_ndcg_5 = ndcg_at_k(true_list=true_list, pred_list=pop_not_bought, k=5)
    pop_ndcg_10 = ndcg_at_k(true_list=true_list, pred_list=pop_not_bought, k=10)
    pop_ndcg_20 = ndcg_at_k(true_list=true_list, pred_list=pop_not_bought, k=20)
    pop_recall_5 = recall_at_k(true_list=true_list, pred_list=pop_not_bought, k=5)
    pop_recall_10 = recall_at_k(true_list=true_list, pred_list=pop_not_bought, k=10)
    pop_recall_20 = recall_at_k(true_list=true_list, pred_list=pop_not_bought, k=20)

    pop_sum_ndcg_5 += pop_ndcg_5
    pop_sum_ndcg_10 += pop_ndcg_10
    pop_sum_ndcg_20 += pop_ndcg_20
    pop_sum_recall_5 += pop_recall_5
    pop_sum_recall_10 += pop_recall_10
    pop_sum_recall_20 += pop_recall_20


pop_ndcg_5 = pop_sum_ndcg_5 / len(dataset.test_seqs)
pop_ndcg_10 = pop_sum_ndcg_10 / len(dataset.test_seqs)
pop_ndcg_20 = pop_sum_ndcg_20 / len(dataset.test_seqs)
pop_recall_5 = pop_sum_recall_5 / len(dataset.test_seqs)
pop_recall_10 = pop_sum_recall_10 / len(dataset.test_seqs)
pop_recall_20 = pop_sum_recall_20 / len(dataset.test_seqs)

print("pop_ndcg_5", pop_ndcg_5)
print("pop_ndcg_10", pop_ndcg_10)
print("pop_ndcg_20", pop_ndcg_20)

print("pop_recall_5", pop_recall_5)
print("pop_recall_10", pop_recall_10)
print("pop_recall_20", pop_recall_20)
