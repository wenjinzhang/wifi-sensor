from flask import Flask, render_template, request, jsonify
app = Flask('__WIFI sensor__')
# a dict list {'start_time': '14:52:27', 'end_time': '', 'inference_time': '', 'act': '', 'user': ''}
activities = [
    # {'start_time': '14:52:27', 'end_time': '14:52:31', 'inference_time': 0.08132, 'act': 'squating', 'user': 'Wenjin'},
]

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/activity", methods=['GET', 'POST'])
def activity():
    activity = request.form.to_dict()
    print(activities)
    if len(activities) > 0 and activities[0]['end_time'] == '':
        del activities[0]
    activities.insert(0, activity)
    return "success"


@app.route("/show_act", methods=['GET'])
def show_act():
    return jsonify(result=activities)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)