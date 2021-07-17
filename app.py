

# col_names = ["status", "duration", "credit_history", "purpose", "credit_amount",
#             "savings_account", "employed_since", "installment_rate", "maritial_status_sex",
#             "other_debtors", "resident_since", "property", "age", "other_installments",
#             "housing", "existing_credits", "job", "no_of_dependents", "telephone", "foreign_worker", "credit"]

# ['status', 'credit_history', 'purpose', 'savings_account',
#        'employed_since', 'maritial_status_sex', 'other_debtors', 'property',
#        'other_installments', 'housing', 'job', 'telephone', 'foreign_worker'],


from flask import Flask, jsonify, request, session, redirect, url_for, g
from flask.templating import render_template
from ml_utils import predict_credit, some_trial
import sys


print('Hello World')

app = Flask(__name__)
app.secret_key = 'jumpjacks'


@app.route('/', methods = ['GET'])
def home():
    return render_template('index.html', message = "Home Message")

    
@app.route('/predict', methods =['GET', 'POST'])
def predict():
    if request.method == 'GET':
        message = "Let us predict"
        return render_template('predict.html', message = message)
    else:
        col_names = ["status", "duration", "credit_history", "purpose", "credit_amount",
            "savings_account", "employed_since", "installment_rate", "maritial_status_sex",
            "other_debtors", "resident_since", "property", "age", "other_installments",
            "housing", "existing_credits", "job", "no_of_dependents", "telephone", "foreign_worker"]
        query_dict = {}
        for name in col_names:
            if name == "duration":
                query_dict[name] = int(request.form[name])
            elif name == "credit_amount":
                query_dict[name] = int(request.form[name])
            elif name == "installment_rate":
                query_dict[name] = int(request.form[name])
            elif name == "resident_since":
                query_dict[name] = int(request.form[name])
            elif name == "age":
                query_dict[name] = int(request.form[name])
            elif name == "existing_credits":
                query_dict[name] = int(request.form[name])
            elif name == "no_of_dependents":
                query_dict[name] = int(request.form[name])
            else:
                query_dict[name] = request.form[name]

        print(query_dict)
        print(some_trial())

        prediction = predict_credit(query_dict)
        message = f'This customer is {prediction} to give loan'
        return render_template('predict.html', message=message)

    # if request.method == 'GET':
    #     message = 'Please sign up!'
    #     return render_template("signup.html")
    # else:
    #     username = request.form["username"]
    #     password = request.form["password"]
    #     favorite_color = request.form["favorite_color"]
    #     message = model.signup(username, password, favorite_color)
    #     return render_template("signup.html", message = message)

if __name__ == '__main__':
    app.run(port=5000, debug=True)