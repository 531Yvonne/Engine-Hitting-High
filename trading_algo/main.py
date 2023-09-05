# region imports
from AlgorithmImports import *
# endregion

from sentiment import WordScore
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


class EngineHittingHigh(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2018, 1, 1)
        self.SetCash(100000)
        # Add Target Companies Stocks
        self.CreateStocksPool()
        self.Schedule.On(self.DateRules.WeekEnd(),
                         self.TimeRules.At(15, 30),
                         self.Rebalance)
        self.ws = WordScore()

    def CreateStocksPool(self):
        # Add Top 20 Automotive Stocks based on total market capitalization as of 6/30/2023
        self.automotive_stocks = ["TSLA", "TM", "BYDDF", "MBGYY", "VWAGY",
                                  "BMWYY", "STLA", "RACE", "HMC", "F",
                                  "GM", "VLVLY", "LI", "HYMTF", "GWLLY",
                                  "NIO", "MAHMF", "RIVN", "SZKMY", "NSANY"]
        self.symbol_objects = {}
        self.news_objects = {}
        for symbol in self.automotive_stocks:
            self.symbol_objects[symbol] = self.AddEquity(
                symbol, Resolution.Daily).Symbol
            self.news_objects[symbol] = self.AddData(
                TiingoNews, self.symbol_objects[symbol]).Symbol

    def Rebalance(self):
        # Calculate most updated indicator scores and ML prediction scores
        indicator_scores, ml_scores = self.CalculateIndicatorScores()

        # Calculate most updated sentiment scores
        sentiment_scores = self.CalculateSentimentScores()

        # Calculate new weights based on the updated scores
        portfolio_weights = self.CalculatePortfolioWeights(
            indicator_scores, sentiment_scores, ml_scores)

        # Liquidate existing positions
        for security in self.Portfolio.Values:
            if security.Invested:
                self.Log(f"Liquidate current holding of {security.Symbol}")
                self.SetHoldings(security.Symbol, 0)

        if portfolio_weights == {}:
            return
        else:
            # Buy new positions based on the updated weights
            for symbol, weight in portfolio_weights.items():
                self.Log(f"{symbol} rebalanced to {weight}")
                self.SetHoldings(symbol, weight)

    def CalculateIndicatorScores(self):
        indicator_scores = {}
        ml_scores = {}
        for (symbol, obj) in self.symbol_objects.items():
            (score, ml_score) = self.CalculateScoreForSymbol(obj)
            if score is not None:
                indicator_scores[symbol] = score
                ml_scores[symbol] = ml_score
        return (indicator_scores, ml_scores)

    def CalculateScoreForSymbol(self, obj):
        data = self.History(obj, 100)
        if data.empty:
            self.Log(f"No historical data available for {obj.Value}")
            return (None, None)
        else:
            # Extract the closing prices from the historical data
            price_data = data["close"]

            # Calculate Exponential Moving Averages (EMA)
            ema_short = price_data.ewm(span=12, adjust=False).mean()
            ema_long = price_data.ewm(span=26, adjust=False).mean()
            ema_avg = (ema_short + ema_long) / 2

            # Calculate Moving Average Convergence Divergence (MACD)
            macd_line = ema_short - ema_long
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            histogram = macd_line - signal_line

            # Calculate Relative Strength Index (RSI)
            delta = price_data.diff()
            gain = delta.where(delta > 0, 0).fillna(0)
            loss = -delta.where(delta < 0, 0).fillna(0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            # Indicators Weights
            ema_weight = 1 / 3
            macd_weight = 1 / 3
            rsi_weight = 1 - (2/3)

            ema_avg = ema_avg.fillna(0)
            histogram = histogram.fillna(0)
            rsi = rsi.fillna(0)

            # Return the final weighted indicator score
            final_score = ema_avg * ema_weight + histogram * macd_weight + rsi * rsi_weight
            self.Log(f"final_score is {final_score}")
            self.Log(f"The final score for {obj.Value} is {final_score[-1]}")

            # Create a DataFrame for features
            features = pd.DataFrame({
                'EMA': ema_avg,
                'MACD_Histogram': histogram,
                'RSI': rsi
            })

            # Prepare the target variable (e.g., price change)
            target = final_score

            # Drop rows with NaN values (due to shifting)
            features = features[:-1]
            target = target[:-1]

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42)

            # Train a Random Forest Regressor
            model = RandomForestRegressor(random_state=42)
            model.fit(X_train, y_train)

            # Make predictions
            predictions = model.predict(X_test)

            # Calculate Mean Absolute Error (MAE)
            mae = mean_absolute_error(y_test, predictions)

            self.Log(f"ML Prediction Mean Absolute Error: {mae}")

            # Use the last data point for prediction as ml_predicted_score
            ml_predicted_score = model.predict(
                features.iloc[-1].values.reshape(1, -1))[0]
            self.Log(
                f"The ML predicted score for {obj.Value} is {ml_predicted_score}")

            return (final_score[-1], ml_predicted_score)

    def CalculateSentimentScores(self):
        sentiment_scores = {}
        for (symbol, obj) in self.symbol_objects.items():
            score = self.CalculateSentimentScoreForSymbol(symbol, obj)
            if score is not None:
                sentiment_scores[symbol] = score
        return sentiment_scores

    def CalculateSentimentScoreForSymbol(self, symbol, obj):
        data = self.History(obj, 5, Resolution.Daily)
        if data.empty:
            self.Log(f"No historical news available for {symbol}")
            return None
        else:
            # Extract the news articles from the historical news
            articles = data.Description
            whole_article = " ".join(articles)
            self.Log(f"Extracted {len(articles)} posts for {symbol}")

            # Return the final sentiment score
            final_score = self.ws.score(whole_article)
            self.Log(f"The final score for {symbol} is {final_score}")
            return final_score

    def CalculatePortfolioWeights(self, indicator_scores={}, sentiment_scores={}, ml_scores={}):

        if not indicator_scores and not sentiment_scores and not ml_scores:
            self.Log("Final Weight doesn't exist!!! Can't allocate!!!")
            return {}
        else:
            # Convert the dictionary to a pandas DataFrame for easier manipulation
            df_indicator = pd.DataFrame(list(indicator_scores.items()), columns=[
                                        'Stock Symbol', 'Indicator Score'])

            df_sentiment = pd.DataFrame(list(sentiment_scores.items()), columns=[
                                        'Stock Symbol', 'Sentiment Score'])
            # Normalize the sentiment scores to a scale of 0 to 1
            min_score = df_sentiment['Sentiment Score'].min()
            max_score = df_sentiment['Sentiment Score'].max()
            df_sentiment['Normalized Score'] = (
                df_sentiment['Sentiment Score'] - min_score) / (max_score - min_score)

            df_ml = pd.DataFrame(list(ml_scores.items()), columns=[
                                 'Stock Symbol', 'ML Score'])
            # Merge DataFrames on 'Stock Symbol' as the common key
            combined_df = df_indicator.merge(
                df_sentiment, on='Stock Symbol').merge(df_ml, on='Stock Symbol')

            # Set a flag to show whether has any scores
            has_score = False

            # Calculate weights based on indicator scores
            total_i_score = combined_df['Indicator Score'].sum()
            self.Log(f"Group Total Indicator Score is {total_i_score}")
            if total_i_score != 0:
                combined_df['Indicator Weight'] = combined_df['Indicator Score'] / total_i_score
                has_score = True
            else:
                combined_df['Indicator Weight'] = 0
                self.Log(
                    "Total Indicator Score for all Stocks is 0!!! Can't allocate!!!")

            # Calculate weights based on sentiment scores
            total_s_score = combined_df['Normalized Score'].sum()
            self.Log(f"Group Total Sentiment Score is {total_s_score}")
            if total_s_score != 0:
                combined_df['Sentiment Weight'] = combined_df['Normalized Score'] / total_s_score
                has_score = True
            else:
                combined_df['Sentiment Weight'] = 0
                self.Log(
                    "Total Sentiment Score for all Stocks is 0!!! Can't allocate!!!")

            # Calculate weights based on ml scores
            total_ml_score = combined_df['ML Score'].sum()
            self.Log(f"Group Total ML Score is {total_ml_score}")
            if total_ml_score != 0:
                combined_df['ML Weight'] = combined_df['ML Score'] / \
                    total_ml_score
                has_score = True
            else:
                combined_df['ML Weight'] = 0
                self.Log("Total ML Score for all Stocks is 0!!! Can't allocate!!!")

            if has_score:
                combined_df['Final Weight'] = (
                    combined_df['Indicator Weight'] + combined_df['Sentiment Weight'] + combined_df['ML Weight']) / 3
                # Create a dictionary of final symbol weights
                portfolio_weights = dict(
                    zip(combined_df['Stock Symbol'], combined_df['Final Weight']))
                return portfolio_weights
            else:
                return {}
