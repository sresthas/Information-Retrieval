from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate

reader = Reader(line_format='user item rating timestamp', sep='::')
data = Dataset.load_from_file('./ml-1m/ratings.dat', reader=reader)

algo = SVD()

cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
