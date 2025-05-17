import pandas as pd
from collections import Counter
import re
import plotly.express as px
import plotly.io as pio

queries_df = pd.read_csv("Queries.csv")
print(queries_df.head())

print(queries_df.info())

queries_df['CTR'] = queries_df['CTR'].str.rstrip('%').astype('float') / 100

def clean_and_split(query):
    words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
    return words

word_counts = Counter()
for query in queries_df['Top queries']:
    word_counts.update(clean_and_split(query))

word_freq_df = pd.DataFrame(word_counts.most_common(20), columns=['Word', 'Frequency'])

fig = px.bar(word_freq_df, x='Word', y='Frequency', title='Top 20 Most Common Words in Search Queries')
fig.show()

top_queries_clicks_vis = queries_df.nlargest(10, 'Clicks')[['Top queries', 'Clicks']]
top_queries_impressions_vis = queries_df.nlargest(10, 'Impressions')[['Top queries', 'Impressions']]

fig_clicks = px.bar(top_queries_clicks_vis, x='Top queries', y='Clicks', title='Top Queries by Clicks')
fig_impressions = px.bar(top_queries_impressions_vis, x='Top queries', y='Impressions', title='Top Querries by Impressions')
fig_clicks.show()
fig_impressions.show()

top_ctr_vis = queries_df.nlargest(10, 'CTR')[['Top queries', 'CTR']]
bottom_ctr_vis = queries_df.nsmallest(10,'CTR')[['Top queries', 'CTR']]

fig_top_ctr = px.bar(top_ctr_vis, x='Top queries', y='CTR', title='Top queries by CTR')
fig_bottom_ctr = px.bar(bottom_ctr_vis, x='Top queries', y='CTR', title='Bottom queries by CTR')
fig_top_ctr.show()
fig_bottom_ctr.show()

correlation_matrix = queries_df[['Clicks', 'Impressions', 'CTR', 'Position']].corr()
fig_corr = px.imshow(correlation_matrix, text_auto=True, title='Correlation Matrix')
fig_corr.show()

from sklearn.ensemble import IsolationForest
features = queries_df[['Clicks', 'Impressions', 'CTR', 'Position']]

iso_forest = IsolationForest(n_estimators=100,contamination=0.01)
iso_forest.fit(features)

queries_df['anomaly'] = iso_forest.predict(features)

anomalies = queries_df[queries_df['anomaly'] == -1]

print(anomalies[['Top queries', 'Clicks', 'Impressions', 'CTR', 'Position']])
