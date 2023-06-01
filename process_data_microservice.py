from flask import Flask, request, send_file
import requests
import pandas as pd
import csv


app = Flask(__name__)


@app.route('/dataset', methods=['POST', 'GET'])
def process():
    try:

        csv_file = request.files['file']

        file_hate = pd.read_csv(csv_file)
        class_data = file_hate['class']
        text = file_hate['tweet']

        length = len(class_data)
        new_text = text.copy()

        for i in range(length):
            line = text[i]
            if (line.find(':') != -1):
                poz = line.find(':')
                tweet = line[poz + 1:]
            else:
                tweet = ''.join(letter for letter in text[i] if letter.isalnum() or letter == ' ')

            new_text[i] = tweet

        with open('dataset.txt', 'w') as file:
            for index in range(length):
                if class_data[index] == 0 or class_data[index] == 1:
                    file.writelines(new_text[index] + ' . ' + ' 1\n')
                else:
                    file.writelines(new_text[index] + ' . ' + ' 0\n')

        return send_file('dataset.txt', mimetype='text/plain', as_attachment=True)

    except Exception as e:
        return str(e), 500


if __name__ == '__main__':
    app.run(port=5003)
