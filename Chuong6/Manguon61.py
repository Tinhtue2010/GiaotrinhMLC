import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

# representation
import matplotlib.pyplot as plt
import seaborn as sns

# pre-processing
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, label_binarize
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# evaluation
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix

# model
import lightgbm as lgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans, DBSCAN
