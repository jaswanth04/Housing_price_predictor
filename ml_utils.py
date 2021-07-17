import pickle


def load_model():
	filename = "model/finalized_model.sav"
	loaded_model = pickle.load(open(filename, 'rb'))
	return loaded_model

def some_trial():
	return "Loaded ml utils"

# function to predict the flower using the model
def predict_credit(query_dict):
	feature_vector = [0] * 61
	feature_index_dict = {'A11': 0, 'A12': 1, 'A13': 2, 'A14': 3, 'A30': 4, 'A31': 5, 'A32': 6, 
		'A33': 7, 'A34': 8, 'A40': 9, 'A41': 10, 'A410': 11, 'A42': 12, 'A43': 13, 'A44': 14, 'A45': 15, 
		'A46': 16, 'A48': 17, 'A49': 18, 'A61': 19, 'A62': 20, 'A63': 21, 'A64': 22, 'A65': 23, 'A71': 24, 
		'A72': 25, 'A73': 26, 'A74': 27, 'A75': 28, 'A91': 29, 'A92': 30, 'A93': 31, 'A94': 32, 'A101': 33, 
		'A102': 34, 'A103': 35, 'A121': 36, 'A122': 37, 'A123': 38, 'A124': 39, 'A141': 40, 'A142': 41, 
		'A143': 42, 'A151': 43, 'A152': 44, 'A153': 45, 'A171': 46, 'A172': 47, 'A173': 48, 'A174': 49, 
		'A191': 50, 'A192': 51, 'A201': 52, 'A202': 53, 'duration': 54, 'credit_amount': 55, 'installment_rate': 56, 
		'resident_since': 57, 'age': 58, 'existing_credits': 59, 'no_of_dependents': 60}

	for feature in query_dict:
		# print(str(type(query_dict[feature])))
		if type(query_dict[feature]) is str:
			feature_vector[feature_index_dict[query_dict[feature]]] = 1
		else:
			feature_vector[feature_index_dict[feature]] = query_dict[feature]

	print(feature_vector)

	model = load_model()

	prediction = model.predict([feature_vector])

	if prediction == 1:
		return "Bad"
	else:
		return "Good"
    # print(best_clf)
    # prediction = best_clf["classifier"].predict([x])[0]
    # print(f"Model prediction: {classes[prediction]}")
    # return classes[prediction]

# function to retrain the model as part of the feedback loop
def retrain(data):
    # pull out the relevant X and y from the FeedbackIn object
    X = [list(d.dict().values())[:-1] for d in data]
    y = [r_classes[d.flower_class] for d in data]

    # fit the classifier again based on the new data obtained
    best_clf["classifier"].fit(X, y)


# sample_query_dict = {'status': 'A12', 'duration': 13, 'credit_history': 'A31', 'purpose': 'A41', 
# 	'credit_amount': 1000, 'savings_account': 'A62', 'employed_since': 'A73', 'installment_rate': 4, 
# 	'maritial_status_sex': 'A92', 'other_debtors': 'A102', 'resident_since': 2, 
# 	'property': 'A122', 'age': 24, 'other_installments': 'A142', 'housing': 'A151', 
# 	'existing_credits': 4, 'job': 'A171', 'no_of_dependents': 2, 'telephone': 'A192', 'foreign_worker': 'A201'}

# predict(sample_query_dict)
