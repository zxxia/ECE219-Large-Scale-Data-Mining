'''
Helper funtions
'''
import json
from typing import List
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer

KICK_OFF_TIME = 1422833400
GAME_OVER_TIME = 1422846360


START_T = [1422833400, 1422835860, 1422837180, 1422838020, 1422838620,
           1422840960, 1422841980, 1422844020, 1422845200]

END_T = [1422835860, 1422837180, 1422838020, 1422838620, 1422840960,
         1422841980, 1422844020, 1422845200, 1422846360]

SCORE_DIFF = [0, 7, 0, 7, 0, -3, -10, -3, 4]

def get_score_diff(timestamp: float) -> int:
    '''
    Get score difference for a given timestamp
    '''
    if timestamp >= GAME_OVER_TIME:
        return 4
    for count, (start, end) in enumerate(zip(START_T, END_T)):
        if timestamp >= start and timestamp < end:
            return SCORE_DIFF[count]

def is_massachusetts(location: str) -> bool:
    '''
    Return True if location is in Massachusetts otherwise return false.
    '''
    return (re.match('.*MA.*', location) is not None or
            re.match('.*Mass.*', location) is not None)

def is_washington(location: str) -> bool:
    '''
    Return True if location is in Washington otherwise return false.
    '''
    return (re.match('.*WA.*', location) is not None or
            re.match('.*Wash.*', location) is not None)

def filter_condition(language: str, timestamp: str, location: str) -> bool:
    '''
    Filtering condition to keep useful tweets.
    '''
    return (language == 'en' and
            float(timestamp) >= 1422833400 and
            (is_massachusetts(location) or is_washington(location)))


def load_tweets(filename: str) -> List:
    '''
    Load tweets which satisfy the filter conditions.
    '''
    tweets = []
    with open(filename, 'r') as tweets_file:
        for line in tweets_file:
            tweet = json.loads(line)
            # only keep tweets in english and citation date
            # later or equal to the opening time of super bowl
            if filter_condition(tweet['tweet']['lang'],
                                tweet['citation_date'],
                                tweet['tweet']['user']['location']):
                tweets.append(tweet)
    return tweets

def get_tweet_polarity(tweet: str) -> str:
    '''
    Get the polarity of a tweet.
    '''
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(tweet)
    if scores['compound'] > 0:
        return 'Positive'
    if scores['compound'] < 0:
        return 'Negative'
    return 'Neutral'


def extract_features(tweet: dict) -> dict:
    '''
    Extract features from tweet
    '''
    features = dict()
    features['timestamp'] = float(tweet['citation_date'])
    location = tweet['tweet']['user']['location']
    features['Massachusetts'] = int(is_massachusetts(location))
    features['Washington'] = int(is_washington(location))
    features['Retweet Count'] = tweet['tweet']['retweet_count']
    #features['Patriots Fan'] =
    #features['Seahawks Fan'] =
    features['Score Diff'] = get_score_diff(features['timestamp'])

    return features
