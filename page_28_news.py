import requests
import pandas as pd
import streamlit as st



league_options = [
    'England Premier',
    'Spain La Liga',
    'Germany Bundesliga', 
    'Italy Serie A',  
    'France Ligue 1',
]

league_url_suffix_dict = {
    'England Premier': 'eng.1',
    'Spain La Liga': 'esp.1',
    'Germany Bundesliga': 'ger.1',
    'Italy Serie A': 'ita.1',
    'France Ligue 1': 'fra.1',
}

def main():

    selected_league = st.sidebar.selectbox("Select a Model", options=league_options, index=0)

    url = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league_url_suffix_dict[selected_league]}/news"

    response = requests.get(url)
    response.raise_for_status()

    data = response.json()

    articles = []

    for article in data["articles"]:
        articles.append({
            "headline": article["headline"],
            "published": article["published"],
            "description": article.get("description"),
            "url": article["links"]["web"]["href"]
        })

    news_df = pd.DataFrame(articles)


    keywords = [
        "transfer",
        "sign",
        "loan",
        "bid",
        "target",
        "deal",
        "contract"
    ]

    transfer_news = news_df[
        news_df["headline"].str.contains("|".join(keywords), case=False, na=False)
    ]

    st.header(f"Transfer News for {selected_league}", divider='blue')
    st.write("")

    for _,row in transfer_news.iterrows():
        formatted = pd.to_datetime(row["published"]).strftime("%d %b %Y, %H:%M UTC")
        st.markdown(
            f"**{row['headline']}** "
            f"<span style='color:#888888; font-size:0.85em;'>"
            f" - Published: {formatted}</span>",
            unsafe_allow_html=True,
        )

        st.write(f"*{row['description']}*", f"[Read more]({row['url']})")
        st.write("---")



if __name__ == '__main__':
    main()