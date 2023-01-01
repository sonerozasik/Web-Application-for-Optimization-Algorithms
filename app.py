# importing Flask and other modules
from flask import Flask, request, render_template
from run import GeneticAlgorithm, SimulatedAnnealing,GWO
import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

# Flask constructor
app = Flask(__name__)  
 
functions=[0,0,0]
algorithms=["","",""]
lbs=[0,0,0]
ubs=[0,0,0]

algorithmParams= [[0]*6,[0]*6,[0]*6]


algorithm=""
algorithm2=""
popsize=0
gennum=0
mutprob=0
crosstype=""
selection=""
dim=0
lb=0
ub=0
lb2=0
ub2=0
initalTemp=0
isArithmetic=0
maxIter = 0
# A decorator used to tell the application
# which URL is associated function
@app.route('/', methods =["GET", "POST"])
def gfg():
    if request.method == "POST":
        global algorithms
        algorithms[0] = request.form.get("algorithm1")
        algorithms[1] = request.form.get("algorithm2")
        algorithms[2] = request.form.get("algorithm3")
        
        global functions
        functions[0] = int(request.form.get("function1"))
        functions[1] = int(request.form.get("function2"))
        functions[2] = int(request.form.get("function3"))

        global lbs
        global ubs
        lbs[0]=int(request.form.get("lb1"))
        ubs[0]=int(request.form.get("ub1"))
        lbs[1]=int(request.form.get("lb2"))
        ubs[1]=int(request.form.get("ub2"))
        lbs[2]=int(request.form.get("lb3"))
        ubs[2]=int(request.form.get("ub3"))
        global algorithmParams
        for i in range(3):           
            print(i)
            if(algorithms[i]=="GA"):
                algorithmParams[i][0]= int(request.form.get("popsize")) 
                algorithmParams[i][1] = int(request.form.get("gennum"))
                algorithmParams[i][2] = int(request.form.get("mutprob"))
                algorithmParams[i][3] = int(request.form.get("crosstype"))
                algorithmParams[i][4] = int(request.form.get("selection"))
                algorithmParams[i][5] = int(request.form.get("dim"))
                print(algorithmParams)
            elif(algorithms[i]=="SA"):
                algorithmParams[i][0]= int(request.form.get("initalTemp")) 
                algorithmParams[i][1] = int(request.form.get("isArithmetic"))
                algorithmParams[i][2] = int(request.form.get("maxIter"))
            elif(algorithms[i]=="GWO"):
                algorithmParams[i][0]= int(request.form.get("popsize2")) 
                algorithmParams[i][1] = int(request.form.get("gennum2"))
                algorithmParams[i][2] = int(request.form.get("decreaseFrom"))
            print(algorithmParams)
        return render_template("img.html")
    return render_template("form.html")
 
@app.route('/plot.png')
def plot_png():
    global algorithm
    fig = create_figure(algorithm)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure(algorithm):
    fig = Figure()
    global algorithmParams
    axis = fig.add_subplot(1, 1, 1)
    xs = range(100) #todo
    sol=[[0]*100,[0]*100,[0]*100]
    i=0
    for i in range(3):
        if algorithms[i] == "GA":
            sol_ = GeneticAlgorithm(functions[i],lbs[i],ubs[i],algorithmParams[i][0],algorithmParams[i][1],algorithmParams[i][2],algorithmParams[i][5],algorithmParams[i][3],algorithmParams[i][4])
            sol_.result.pop(0)
            sol[i] = sol_.result
        elif algorithms[i] == "SA":
            sol[i] = SimulatedAnnealing(functions[i],algorithmParams[i][0],algorithmParams[i][1],algorithmParams[i][2])
        elif algorithms[i] == "GWO":
            sol[i] = GWO(functions[i],lbs[3],ubs[3],30,algorithmParams[i][0],algorithmParams[i][1],algorithmParams[i][3])

    axis.plot(xs, sol[0],color='r', label=algorithms[0])
    axis.plot(xs, sol[1],color='y', label=algorithms[1])
    axis.plot(xs, sol[2],color='b', label=algorithms[2])
    axis.legend()
    return fig

if __name__=='__main__':
   app.run()






