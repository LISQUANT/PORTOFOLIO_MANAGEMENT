from src.get_data.data_handler import DataHandler
from src.HMM.detector.features import FeatureMatrix
from src.HMM.detector.labler import RegimeLabeler
from src.HMM.detector.regime_detector import RegimeDetector
from src.HMM.visual.regime_visual import RegimeVisualizer
from src.HMM.backtest import Backtester

portfolio_to_analyse = {'AMZN': 10, 'ASML': 3, 'ELF': 30, 'GOOGL': 17, 'HOOD': 30, 'URTH':85, 'TSLA': 8, 'UBER': 20, 'UNH': 12}

data     = DataHandler(tickers=portfolio_to_analyse)
features = FeatureMatrix(data_handler=data)
detector = RegimeDetector(feature_matrix=features)
labeler  = RegimeLabeler(detector=detector, feature_matrix=features)

bt = Backtester(data, features, detector, labeler)

for ticker in portfolio_to_analyse:
    bt.summary(ticker)
    bt.plot(ticker)