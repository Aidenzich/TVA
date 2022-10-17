#%%
import pandas as pd


df = pd.read_csv("carrefour_sales.csv")
df = df[["customer", "product", "order_date", "quantity"]]


# %%
df["product"] = df.apply(lambda x: [x["product"]], axis=1)

#%%
df
#%%
gdf = df.groupby(["customer", "order_date"], group_keys=False).agg({"product": "sum"})

# %%
gdf["product"] = gdf.apply(lambda x: ",".join(set(x["product"])), axis=1)

#%%
gdf["quantity"] = 1
gdf.to_csv("carrefour_sales_haddled.csv", index=False)


#%%
gdf["product"].value_counts()

#%%
gdf.reset_index().to_csv("carrefour_sales_haddled.csv", index=False)

#%%
len(gdf['product'].value_counts())

#%%
top100000 = gdf['product'].value_counts()[:100000].index

count1 = 0
for i in top100000:
    if "," not in i:
        count1 += 1
    
print(count1)
    