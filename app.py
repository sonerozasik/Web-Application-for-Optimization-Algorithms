# importing Flask and other modules
from flask import Flask, request, render_template
from run import GeneticAlgorithm, SimulatedAnnealing,GWO,HillClimbing,HarmonySearch
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

algorithmParams= [[0]*7,[0]*7,[0]*7]
gennum =0
# A decorator used to tell the application
# which URL is associated function
@app.route('/', methods =["GET", "POST"])
def gfg():
    if request.method == "POST":
        global algorithms
        global functions
        global lbs
        global ubs
        global gennum
        algorithms[0] = request.form.get("algorithm1")
        algorithms[1] = request.form.get("algorithm2")
        algorithms[2] = request.form.get("algorithm3")

        global algorithmParams
        for i in range(3):           
            if(algorithms[i]!=None):
                print(algorithms[i])
                functions[i] = int(request.form.get("function"+str(i+1)))
                lbs[i]=int(request.form.get("lb"+str(i+1)))
                ubs[i]=int(request.form.get("ub"+str(i+1)))
            if(algorithms[i]=="GA"):
                algorithmParams[i][0]= int(request.form.get("popsize")) 
                algorithmParams[i][1] = int(request.form.get("gennum"))
                gennum = algorithmParams[i][1]
                algorithmParams[i][2] = float(request.form.get("mutprob"))
                algorithmParams[i][3] = int(request.form.get("crosstype"))
                algorithmParams[i][4] = int(request.form.get("selection"))
                algorithmParams[i][5] = int(request.form.get("dim"))
            elif(algorithms[i]=="SA"):
                algorithmParams[i][0]= int(request.form.get("initalTemp")) 
                algorithmParams[i][1] = int(request.form.get("isArithmetic"))
                algorithmParams[i][2] = int(request.form.get("maxIter"))
                gennum = algorithmParams[i][2]
            elif(algorithms[i]=="GWO"):
                algorithmParams[i][0]= int(request.form.get("popsize2")) 
                algorithmParams[i][1] = int(request.form.get("gennum2"))
                algorithmParams[i][2] = int(request.form.get("decreaseFrom"))
                algorithmParams[i][3] = int(request.form.get("dim2"))
                gennum = algorithmParams[i][1]
            elif(algorithms[i]=="HC"):
                algorithmParams[i][0]= int(request.form.get("dim3")) 
                algorithmParams[i][1] = int(request.form.get("gennum3"))
                algorithmParams[i][2] = float(request.form.get("stepsize"))
                gennum = algorithmParams[i][1]
            elif(algorithms[i]=="HS"):
                algorithmParams[i][0]= int(request.form.get("hms")) 
                algorithmParams[i][1] = float(request.form.get("hmcr"))
                algorithmParams[i][2] = float(request.form.get("par"))
                algorithmParams[i][3] = float(request.form.get("bw"))
                algorithmParams[i][4] = int(request.form.get("dim4"))
                algorithmParams[i][5] = int(request.form.get("numproc"))
                algorithmParams[i][6] = int(request.form.get("numiter"))
                gennum = algorithmParams[i][6]


        return render_template("img.html")
    return render_template("form.html")
 
@app.route('/plot.png')
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

def create_figure():
    fig = Figure()
    global algorithmParams
    axis = fig.add_subplot(1, 1, 1)
    xs = range(gennum)
    sol=[[0]*gennum,[0]*gennum,[0]*gennum]
    i=0
    for i in range(3):
        if algorithms[i] == "GA":
            sol_ = GeneticAlgorithm(functions[i],lbs[i],ubs[i],algorithmParams[i][0],algorithmParams[i][1],algorithmParams[i][2],algorithmParams[i][5],algorithmParams[i][3],algorithmParams[i][4])
            sol_.result.pop(0)
            sol[i] = sol_.result
        elif algorithms[i] == "SA":
            sol[i] = SimulatedAnnealing(functions[i],algorithmParams[i][0],algorithmParams[i][1],algorithmParams[i][2])
        elif algorithms[i] == "GWO":
            sol_ = GWO(functions[i],lbs[i],ubs[i],algorithmParams[i][3],algorithmParams[i][0],algorithmParams[i][1],algorithmParams[i][2])
            sol_.result.pop(0)
            sol[i] = sol_.result
        elif algorithms[i] == "HC":
            sol[i] = HillClimbing(functions[i],lbs[i],ubs[i],algorithmParams[i][0],algorithmParams[i][1],algorithmParams[i][2])
        elif algorithms[i] == "HS":
            sol[i] = HarmonySearch(functions[i],lbs[i],ubs[i],algorithmParams[i][0],algorithmParams[i][1],algorithmParams[i][2],algorithmParams[i][3],algorithmParams[i][4],algorithmParams[i][5],algorithmParams[i][6])
    

    if(algorithms[0]!=None):
        axis.plot(xs, sol[0],color='r', label=algorithms[0])
    if(algorithms[1]!=None):
        axis.plot(xs, sol[1],color='y', label=algorithms[1])
    if(algorithms[2]!=None):
        axis.plot(xs, sol[2],color='b', label=algorithms[2])
    
    axis.legend()
    return fig

if __name__=='__main__':
   app.run()






