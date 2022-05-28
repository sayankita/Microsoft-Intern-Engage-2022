import tweepy
import configparser
import pandas as pd



api_key = "RKV3CkQV8sFY3EAvWhH1dyVhn"
api_key_secret = "pQAJfDKqAakprejDaJJ4cHkq6GIERQrFebQN2cuCXNw9CR8XpZ"
access_token = '1527352839391309824-4MitzoB58PCYGg1obvmvtFnMupLU6q'
access_token_secret = "JYzqcBYnl3XVO8TLBlem1eXfSFm48biFzB6ZRu5DsWerv"


# authentication
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

keywords = 'Toyota'
limit=300

tweets = tweepy.Cursor(api.search_tweets, q=keywords, count=100, tweet_mode='extended').items(limit)

# tweets = api.user_timeline(screen_name=user, count=limit, tweet_mode='extended')

# create DataFrame
columns = ['User', 'Tweet']
data = []

for tweet in tweets:
    data.append([tweet.user.screen_name, tweet.full_text])

df = pd.DataFrame(data, columns=columns)

print(df)
