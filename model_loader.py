from keras.models import load_model
import numpy as np
import pandas as pd
from dataloader import get_test
test = 117

test_dir = pd.read_table("./data/sample.csv",sep = ",")['name']
my_model = load_model("densebest2.h5")
test_name = np.array(test_dir).reshape(test)
test_data = get_test()

predict = my_model.predict(test_data)[:, 1]
predicted = np.array(predict).reshape(test)
test_dict = {'Id': test_name, 'Predicted': predicted}

result = pd.DataFrame(test_dict, index=[0 for _ in range(test)])

result.to_csv("reconstruct_result.csv", index=False, sep=',')