import requests
from bs4 import BeautifulSoup
import praw
import json
import datetime
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import psycopg2
from psycopg2.extras import execute_batch
import re
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from GoogleNews import GoogleNews
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sentiment_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("stock_sentiment")

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')

class SentimentAnalyzer:
    def __init__(self, db_config):
        """Initialize the sentiment analyzer with database configuration."""
        self.db_config = db_config
        self.sia = SentimentIntensityAnalyzer()
        
        # Add finance-specific words to VADER lexicon
        self.sia.lexicon.update({
            'bullish': 3.0,
            'bearish': -3.0,
            'outperform': 2.5,
            'underperform': -2.5,
            'buy': 2.0,
            'sell': -2.0,
            'hold': 0.0,
            'upgrade': 2.0,
            'downgrade': -2.0,
            'beat': 1.5,
            'miss': -1.5,
            'profit': 1.5,
            'loss': -1.5,
            'growth': 1.5,
            'decline': -1.5,
            'surge': 2.0,
            'plunge': -2.0,
            'rallied': 1.8,
            'crashed': -1.8,
            'bankruptcy': -3.0,
            'acquisition': 1.0,
            'merger': 1.0,
            'dividend': 1.0,
            'investigation': -1.5,
            'fraud': -3.0,
            'scandal': -2.5,
            'lawsuit': -2.0,
            'settlement': 0.5
        })
        
        # Store NSE symbols for stock matching
        self.nse_symbols = self._load_nse_symbols()
    
    def _load_nse_symbols(self):
        """Load NSE symbols from database."""
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT symbol, company_name FROM companies")
            symbols = {row[0]: row[1] for row in cursor.fetchall()}
            cursor.close()
            conn.close()
            
            if not symbols:
                logger.warning("No symbols found in database. Using dummy data.")
                # Return dummy data for testing if database is empty
                return {"RELIANCE": "Reliance Industries Ltd.", "TCS": "Tata Consultancy Services Ltd."}
            
            return symbols
        except Exception as e:
            logger.error(f"Error loading NSE symbols: {e}")
            return {"RELIANCE": "Reliance Industries Ltd.", "TCS": "Tata Consultancy Services Ltd."}

    def _get_db_connection(self):
        """Create and return a database connection."""
        return psycopg2.connect(**self.db_config)

    def analyze_text(self, text):
        """Analyze sentiment of text using VADER."""
        if not text or not isinstance(text, str):
            return {
                'polarity': 0,
                'subjectivity': 0
            }
            
        sentiment = self.sia.polarity_scores(text)
        
        # Calculate subjectivity as absolute normalized polarity
        # More extreme scores (positive or negative) indicate higher subjectivity
        subjectivity = abs(sentiment['compound'])
        
        return {
            'polarity': sentiment['compound'],  # Range: -1 to 1
            'subjectivity': subjectivity        # Range: 0 to 1
        }
    
    def match_stock_symbols(self, text):
        """Find any NSE stock symbols mentioned in the text."""
        found_symbols = []
        
        # Direct symbol matches
        for symbol in self.nse_symbols.keys():
            pattern = r'\b' + re.escape(symbol) + r'\b'
            if re.search(pattern, text):
                found_symbols.append(symbol)
        
        # Company name matches
        for symbol, company_name in self.nse_symbols.items():
            if company_name and company_name.lower() in text.lower():
                if symbol not in found_symbols:
                    found_symbols.append(symbol)
        
        return found_symbols
    
    def store_social_media_sentiment(self, data):
        """Store social media sentiment data in the database."""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        insert_query = """
        INSERT INTO social_media_sentiment 
        (symbol, source, title, url, post_date, collected_date, score, sentiment_polarity, sentiment_subjectivity)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        """
        
        try:
            execute_batch(cursor, insert_query, data)
            conn.commit()
            logger.info(f"Successfully stored {len(data)} social media sentiment entries")
        except Exception as e:
            logger.error(f"Error storing social media sentiment: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()
    
    def store_news_sentiment(self, data):
        """Store news sentiment data in the database."""
        conn = self._get_db_connection()
        cursor = conn.cursor()
        
        insert_query = """
        INSERT INTO news_sentiment 
        (symbol, source, headline, url, published_date, collected_date, sentiment_polarity, sentiment_subjectivity, news_type)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        """
        
        try:
            execute_batch(cursor, insert_query, data)
            conn.commit()
            logger.info(f"Successfully stored {len(data)} news sentiment entries")
        except Exception as e:
            logger.error(f"Error storing news sentiment: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()


class RedditScraper:
    def __init__(self, client_id, client_secret, user_agent):
        """Initialize Reddit scraper with API credentials."""
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        
        # Subreddits to monitor
        self.subreddits = [
            "IndianStockMarket", 
            "StockMarketIndia", 
            "stocks", 
            "IndiaInvestments", 
            "investing", 
            "NSEIndia",
            "DalalStreetTalks"
        ]
    
    def scrape(self, limit=50, time_filter="day"):
        """Scrape posts from stock-related subreddits."""
        posts = []
        
        for subreddit_name in self.subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Get top posts from today
                for submission in subreddit.top(time_filter=time_filter, limit=limit):
                    post_data = {
                        "title": submission.title,
                        "selftext": submission.selftext,
                        "score": submission.score,
                        "url": submission.url,
                        "permalink": f"https://www.reddit.com{submission.permalink}",
                        "created_utc": datetime.datetime.fromtimestamp(submission.created_utc),
                        "subreddit": subreddit_name,
                        "num_comments": submission.num_comments
                    }
                    posts.append(post_data)
                
                # Get hot posts
                for submission in subreddit.hot(limit=limit):
                    post_data = {
                        "title": submission.title,
                        "selftext": submission.selftext,
                        "score": submission.score,
                        "url": submission.url,
                        "permalink": f"https://www.reddit.com{submission.permalink}",
                        "created_utc": datetime.datetime.fromtimestamp(submission.created_utc),
                        "subreddit": subreddit_name,
                        "num_comments": submission.num_comments
                    }
                    posts.append(post_data)
                
                logger.info(f"Successfully scraped {subreddit_name}")
            except Exception as e:
                logger.error(f"Error scraping subreddit {subreddit_name}: {e}")
        
        # Remove duplicates based on permalink
        unique_posts = {post["permalink"]: post for post in posts}
        return list(unique_posts.values())


class NewsScraper:
    def __init__(self):
        """Initialize news scraper."""
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    
    def scrape_moneycontrol(self):
        """Scrape news from Moneycontrol."""
        urls = [
            "https://www.moneycontrol.com/news/business/markets/",
            "https://www.moneycontrol.com/news/business/stocks/",
            "https://www.moneycontrol.com/news/business/earnings/"
        ]
        
        all_articles = []
        
        for url in urls:
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                if response.status_code != 200:
                    logger.warning(f"Failed to retrieve data from {url}, status code: {response.status_code}")
                    continue
                
                soup = BeautifulSoup(response.text, "html.parser")
                articles = []
                
                # Parse articles from the page
                for article in soup.find_all("li", class_="clearfix"):
                    title_element = article.find("h2")
                    link_element = article.find("a")
                    time_element = article.find("span", class_="article_date")
                    
                    if title_element and link_element:
                        title = title_element.text.strip()
                        url = link_element["href"]
                        
                        # Extract date if available
                        published_date = None
                        if time_element:
                            date_text = time_element.text.strip()
                            try:
                                published_date = datetime.datetime.strptime(date_text, "%b %d, %Y %I:%M %p IST")
                            except:
                                published_date = datetime.datetime.now()
                        else:
                            published_date = datetime.datetime.now()
                        
                        articles.append({
                            "title": title,
                            "url": url,
                            "source": "Moneycontrol",
                            "published_date": published_date,
                            "news_type": "General"
                        })
                
                all_articles.extend(articles)
                logger.info(f"Scraped {len(articles)} articles from {url}")
            
            except Exception as e:
                logger.error(f"Error scraping Moneycontrol {url}: {e}")
        
        return all_articles
    
    def scrape_economic_times(self):
        """Scrape news from Economic Times."""
        urls = [
            "https://economictimes.indiatimes.com/markets/stocks/news",
            "https://economictimes.indiatimes.com/markets/stocks/earnings",
            "https://economictimes.indiatimes.com/markets/stocks/recos"
        ]
        
        all_articles = []
        
        for url in urls:
            try:
                response = requests.get(url, headers=self.headers, timeout=10)
                if response.status_code != 200:
                    logger.warning(f"Failed to retrieve data from {url}, status code: {response.status_code}")
                    continue
                
                soup = BeautifulSoup(response.text, "html.parser")
                articles = []
                
                # Parse articles from the page
                for article in soup.find_all("div", class_="eachStory"):
                    title_element = article.find("h3")
                    if not title_element:
                        continue
                        
                    link_element = title_element.find("a")
                    if not link_element:
                        continue
                    
                    title = title_element.text.strip()
                    url = "https://economictimes.indiatimes.com" + link_element["href"] if link_element["href"].startswith("/") else link_element["href"]
                    
                    # Try to find date
                    date_element = article.find("time")
                    published_date = datetime.datetime.now()
                    if date_element and date_element.get("datetime"):
                        try:
                            published_date = datetime.datetime.strptime(date_element["datetime"], "%Y-%m-%d %H:%M:%S")
                        except:
                            pass
                    
                    articles.append({
                        "title": title,
                        "url": url,
                        "source": "Economic Times",
                        "published_date": published_date,
                        "news_type": "General"
                    })
                
                all_articles.extend(articles)
                logger.info(f"Scraped {len(articles)} articles from {url}")
            
            except Exception as e:
                logger.error(f"Error scraping Economic Times {url}: {e}")
        
        return all_articles
    
    def scrape_google_news(self, query, news_type="General", days=2):
        """Scrape news from Google News API."""
        try:
            gn = GoogleNews(lang='en', region='IN', period=f'{days}d')
            gn.search(query)
            results = gn.results()
            
            articles = []
            for item in results:
                try:
                    # Convert date
                    if 'datetime' in item and item['datetime']:
                        published_date = item['datetime']
                    else:
                        # If no date, use current date minus random hours to distribute
                        hours_ago = (hash(item['title']) % 48) if 'title' in item else 24
                        published_date = datetime.datetime.now() - datetime.timedelta(hours=hours_ago)
                    
                    articles.append({
                        "title": item.get('title', ''),
                        "url": item.get('link', ''),
                        "source": item.get('site', 'Google News'),
                        "published_date": published_date,
                        "news_type": news_type
                    })
                except Exception as e:
                    logger.error(f"Error processing Google News item: {e}")
            
            logger.info(f"Scraped {len(articles)} articles from Google News for query '{query}'")
            return articles
            
        except Exception as e:
            logger.error(f"Error scraping Google News for query '{query}': {e}")
            return []


def process_social_media(analyzer, reddit_scraper):
    """Process social media data and store sentiment analysis."""
    posts = reddit_scraper.scrape()
    sentiment_data = []
    
    for post in posts:
        combined_text = f"{post['title']} {post['selftext']}"
        sentiment = analyzer.analyze_text(combined_text)
        
        # Match stock symbols
        symbols = analyzer.match_stock_symbols(combined_text)
        
        if symbols:
            # If specific stocks are mentioned, create entries for each
            for symbol in symbols:
                sentiment_entry = (
                    symbol,                                      # symbol
                    f"Reddit/{post['subreddit']}",               # source
                    post['title'],                               # title
                    post['permalink'],                           # url
                    post['created_utc'],                         # post_date
                    datetime.datetime.now(),                     # collected_date
                    post['score'],                               # score
                    sentiment['polarity'],                       # sentiment_polarity
                    sentiment['subjectivity']                    # sentiment_subjectivity
                )
                sentiment_data.append(sentiment_entry)
        else:
            # General market sentiment (no specific stock)
            sentiment_entry = (
                None,                                           # symbol (NULL)
                f"Reddit/{post['subreddit']}",                  # source
                post['title'],                                  # title
                post['permalink'],                              # url
                post['created_utc'],                            # post_date
                datetime.datetime.now(),                        # collected_date
                post['score'],                                  # score
                sentiment['polarity'],                          # sentiment_polarity
                sentiment['subjectivity']                       # sentiment_subjectivity
            )
            sentiment_data.append(sentiment_entry)
    
    # Store in database
    if sentiment_data:
        analyzer.store_social_media_sentiment(sentiment_data)
    
    return len(sentiment_data)


def process_news(analyzer, news_scraper):
    """Process news data and store sentiment analysis."""
    # Get news from different sources
    moneycontrol_news = news_scraper.scrape_moneycontrol()
    economic_times_news = news_scraper.scrape_economic_times()
    
    # Get geopolitical news that might affect markets
    geopolitical_news = news_scraper.scrape_google_news(
        "trade tariffs OR sanctions OR geopolitical events affecting Indian markets",
        news_type="Geopolitical"
    )
    
    # Get supply chain news
    supply_chain_news = news_scraper.scrape_google_news(
        "supply chain disruption OR logistics issues OR manufacturing India",
        news_type="Supply Chain"
    )
    
    # Combine all news
    all_news = moneycontrol_news + economic_times_news + geopolitical_news + supply_chain_news
    
    sentiment_data = []
    
    for article in all_news:
        sentiment = analyzer.analyze_text(article['title'])
        
        # Match stock symbols
        symbols = analyzer.match_stock_symbols(article['title'])
        
        if symbols:
            # If specific stocks are mentioned, create entries for each
            for symbol in symbols:
                sentiment_entry = (
                    symbol,                                      # symbol
                    article['source'],                           # source
                    article['title'],                            # headline
                    article['url'],                              # url
                    article['published_date'],                   # published_date
                    datetime.datetime.now(),                     # collected_date
                    sentiment['polarity'],                       # sentiment_polarity
                    sentiment['subjectivity'],                   # sentiment_subjectivity
                    article['news_type']                         # news_type
                )
                sentiment_data.append(sentiment_entry)
        else:
            # General market news (no specific stock)
            sentiment_entry = (
                None,                                           # symbol (NULL)
                article['source'],                              # source
                article['title'],                               # headline
                article['url'],                                 # url
                article['published_date'],                      # published_date
                datetime.datetime.now(),                        # collected_date
                sentiment['polarity'],                          # sentiment_polarity
                sentiment['subjectivity'],                      # sentiment_subjectivity
                article['news_type']                            # news_type
            )
            sentiment_data.append(sentiment_entry)
    
    # Store in database
    if sentiment_data:
        analyzer.store_news_sentiment(sentiment_data)
    
    return len(sentiment_data)


def main():
    parser = argparse.ArgumentParser(description='Stock Sentiment Analysis for NSE')
    parser.add_argument('--reddit-client-id', default="PVs4-U0vxQ56XsGDH6vZQg", help='Reddit API client ID')
    parser.add_argument('--reddit-client-secret', default="wTEdNfP1YRGlW4g-t7uXS5xhzb3y5A", help='Reddit API client secret')
    parser.add_argument('--db-host', default='localhost', help='Database host')
    parser.add_argument('--db-port', default=5432, type=int, help='Database port')
    parser.add_argument('--db-name', default='stock_profiler', help='Database name')
    parser.add_argument('--db-user', default='postgres', help='Database user')
    parser.add_argument('--db-password', default='postgres', help='Database password')
    
    args = parser.parse_args()
    
    # Database configuration
    db_config = {
        'host': args.db_host,
        'port': args.db_port,
        'database': args.db_name,
        'user': args.db_user,
        'password': args.db_password
    }
    
    # Initialize components
    analyzer = SentimentAnalyzer(db_config)
    reddit_scraper = RedditScraper(
        client_id=args.reddit_client_id,
        client_secret=args.reddit_client_secret,
        user_agent="StockSentimentScraper by u/ApprehensiveDonut463"
    )
    news_scraper = NewsScraper()
    
    logger.info("Starting sentiment analysis process")
    
    # Process social media
    social_media_count = process_social_media(analyzer, reddit_scraper)
    logger.info(f"Processed {social_media_count} social media posts")
    
    # Process news
    news_count = process_news(analyzer, news_scraper)
    logger.info(f"Processed {news_count} news articles")
    
    logger.info("Sentiment analysis process completed")


if __name__ == "__main__":
    main()