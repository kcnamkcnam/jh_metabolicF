from flask import request, session, redirect, url_for, make_response, render_template, send_file
from app import app
from processing import do_addition, get_mode, process_data, process_input, imputeLabelsFromScratched, predictFluxesFromLabels, add_dropdown_menu
import os

Home_dir = os.getcwd()
App_dir = os.path.dirname(app.instance_path)
# disable debug for production
app.config["DEBUG"] = True
app.config["SECRET_KEY"] = "aoejfakcjefwejanavvnakdaeofwvoiejqporafda"

#def clear_session():
#    session['last_action'] = None
    # using session.clear() nulls everything, including the session itself, so you have to check for session AND session['key'] or pop(None) individual session keys
    # session.clear()

# Check credentials, modify session, etc.
#@app.before_request
#    if 'session_start' not in session:
#        session['session_start'] = datetime.datetime.now()
#    session['last_action'] = datetime.datetime.now()

comments = []
@app.route('/')
@app.route('/index', methods=["GET", "POST"])
def index():
    os.chdir(Home_dir)
    model_dic = add_dropdown_menu(App_dir + "/app/templates/") # add more models to the dropdown menu
    if request.method == 'POST':
        # get input data (file, sheets, model_type)
        input_file = request.files["input_file"]
        sheets_str = request.form.get("sheets")
        model = request.form.get("dropdown")
        model_type = model_dic.get(model) # get model_type from model_dictionaly using model as key.

        result = process_input(input_file, sheets_str) ###take the result variable as the input for ML models###
        labels = imputeLabelsFromScratched(result, model_type)
        fluxes = predictFluxesFromLabels(labels, model_type)

        #print the resulting fluxes in csv
        resultString='\n'.join([str(i) for i in fluxes])
        response = make_response(resultString)
        response.headers["Content-Disposition"] = "attachment; filename=result.csv"
        return response
        # return send_file(file_path, as_attachment=True) # download the result file to the user.
        
        # return '''
        # <html>
        #     <body>
        #         <p>test</p>
        #         {result_file}
        #     </body>
        # </html>
        # '''.format(result_file=result)
    
    if os.path.exists(App_dir + "/app/templates/index_new.html"):
      return render_template("index_new.html") # index_new has more models added to the dropdown menu.
    else:
      return render_template("index.html")

@app.route("/testcomments", methods=["GET", "POST"])
def comments_page():
    if request.method == "GET":
        return render_template("testcomments.html", comments=comments)
    comments.append(request.form["contents"])
    return redirect(url_for('comments_page'))

@app.route('/testadd', methods=['GET', 'POST'])
def adder_page():
    errors = ""
    if request.method == 'POST':
        number1 = None
        number2 = None
        try:
            number1 = float(request.form["number1"])
        except:
            errors += "<p>{!r} is not a number.</p>\n".format(request.form["number1"])
        try:
            number2 = float(request.form["number2"])
        except:
            errors += "<p>{!r} is not a number.</p>\n".format(request.form["number2"])
        if number1 is not None and number2 is not None:
            result = do_addition(number1, number2)
            return '''
                <html>
                    <head>
                        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-GLhlTQ8iRABdZLl6O3oVMWSktQOp6b7In1Zl3/Jr59b6EGGoI1aFkw7cmDA6j6gD" crossorigin="anonymous">
                    </head>
                    <body style="background-color: rgb(236, 236, 226); padding: 30px 30px;">
                        <p>The result is {result}</p>
                        <p><a href="/testadd">Click here to calculate again</a>
                        <div>
                            <a href="/">Home</a>
                        </div>
                    </body>
                </html>
            '''.format(result=result)
    return '''
        <html>
            <head>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-GLhlTQ8iRABdZLl6O3oVMWSktQOp6b7In1Zl3/Jr59b6EGGoI1aFkw7cmDA6j6gD" crossorigin="anonymous">
            </head>
            <body style="background-color: rgb(236, 236, 226); padding: 30px 30px;">
                {errors}
                <form method="post" action="/testadd" novalidate>
                    <label class="form-label">Enter Your Numbers:</label>
                    <div class="mb-3">
                        <input type="number" class="form-control-sm" name="number1">
                    </div>
                    <div class="mb-3">
                        <input type="number" class="form-control-sm" name="number2">
                    </div>
                    <button type="submit" class="btn btn-success">Do Calculation</button>
                </form>
                <div>
                    <a href="/">Home</a>
                </div>
            </body>
        </html>
    '''.format(errors=errors)

@app.route('/testmode', methods=['GET', 'POST'])
def mode_page():
    if "inputs" not in session:
        session["inputs"] = []

    errors = ""
    if request.method == 'POST':
        try:
            session["inputs"].append(float(request.form["number"]))
            session.modified = True
        except:
            errors += "<p>{!r} is not a number.</p>\n".format(request.form["number"])

        if request.form["action"] == "Calculate number":
            result = get_mode(session["inputs"])
            session.clear()
            session.modified = True
            return '''
                <html>
                    <body>
                        <p>{result}</p>
                        <p><a href="/testmode">Click here to calculate again</a>
                    </body>
                </html>
            '''.format(result=result)

    if len(session["inputs"]) == 0:
        numbers_so_far = ""
    else:
        numbers_so_far = "<p>Numbers so far:</p>"
        for number in session["inputs"]:
            numbers_so_far += "<p>{}</p>".format(number)

    return '''
        <html>
            <body>
                {numbers_so_far}
                {errors}
                <p>Enter your number:</p>
                <form method="post" action="/testmode">
                    <p><input name="number" /></p>
                    <p><input type="submit" name="action" value="Add another" /></p>
                    <p><input type="submit" name="action" value="Calculate number" /></p>
                </form>
            </body>
        </html>
    '''.format(numbers_so_far=numbers_so_far, errors=errors)

@app.route("/testsum", methods=['GET', 'POST'])
def file_summer_page():
    if request.method == 'POST':
        input_file = request.files["input_file"]
        input_data = input_file.stream.read().decode("utf-8")
        output_data = process_data(input_data)
        response = make_response(output_data)
        response.headers["Content-Disposition"] = "attachment; filename=result.csv"
        return response

    return '''
        <html>
            <body>
                <p>Select the file you want to sum up:</p>
                <form method="post" action="/testsum" enctype="multipart/form-data">
                    <p><input type="file" name="input_file" /></p>
                    <p><input type="submit" value="Process the file" /></p>
                </form>
            </body>
        </html>
    '''
