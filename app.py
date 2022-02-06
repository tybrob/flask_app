from flask import Flask,render_template,request
import ml
import datetime

app = Flask(__name__)


@app.route('/test', methods=['GET','POST'])
def test():
    data = {}
    if request.form:
        form_data = request.form
        data['form'] = form_data
        print(form_data)
        news_for_prediction = form_data['news_for_prediction']
        model = form_data['model']
        start = datetime.datetime.now()
        if (model == 'nb'):
            prediction = ml.nb_predict([news_for_prediction])
        elif (model == 'svm'):
            prediction = ml.svm_predict([news_for_prediction])
        elif (model == 'lr'):
            prediction = ml.lr_predict([news_for_prediction])

        if (prediction):
            data['prediction'] = 'REAL'
        else:
            data['prediction'] = 'FAKE'

        data['time_to_predict'] = datetime.datetime.now() - start

    return render_template('test.html', data=data)


@app.route('/train', methods=['GET'])
def train():
    data = {} 
    
    start = datetime.datetime.now()
    X_train,X_test,Y_train,Y_test = ml.data_preprocessing()
    data['time_to_process'] = datetime.datetime.now() - start
    start = datetime.datetime.now()
    data['naive_bayes_accuracy'] = ml.nb_train_and_evaluate(X_train,X_test,Y_train,Y_test)
    data['naive_bayes_time_to_train'] = datetime.datetime.now() - start

    start = datetime.datetime.now()
    data['svm_accuracy'] = ml.svm_train_and_evaluate(X_train,X_test,Y_train,Y_test)
    data['svm_time_to_train'] = datetime.datetime.now() - start    

    start = datetime.datetime.now()
    data['lr_accuracy'] = ml.lr_train_and_evaluate(X_train,X_test,Y_train,Y_test)
    data['lr_time_to_train'] = datetime.datetime.now() - start  

    return render_template('train.html', data=data)


if __name__ == '__main__':
    app.run(port = 6789, debug = True)