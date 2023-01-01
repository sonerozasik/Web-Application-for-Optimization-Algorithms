from multiprocessing.resource_sharer import stop
from pyharmonysearch import ObjectiveFunctionInterface, harmony_search
from math import pow
import random
import array
array.typecodes
from multiprocessing import cpu_count
import functions as functions
from enumFunctions import Functions
import GA
from SA import simulated_annealing
from GWO import GWO
from numpy import asarray
from numpy.random import seed
from HC2 import hillclimbing

def GeneticAlgorithm(function,lb,ub,popsize,genNum,mp,dim,crossoverType,selectionMethod):
    obj_func = functions.selectFunction(function)
    sol = GA.GA(obj_func, lb, ub, dim, popsize, genNum,mp,crossoverType,selectionMethod)
    return sol

def SimulatedAnnealing(function,initalTemperature,isArithmetic_,maxIter):
    obj_func = functions.selectFunction(function)
    # dim array size, -5 lb +5 lb 
    sol =simulated_annealing( min_values = [getBounds(function)[0]]*30, max_values = [getBounds(function)[1]]*30, mu = 0, sigma = 1, initial_temperature = initalTemperature,temperature_iterations=maxIter ,
        final_temperature = 0.0001, alpha = 0.5, target_function = obj_func, verbose = True, isArithmetic=isArithmetic_)
    return sol

def gwo(function,popSize,numOfGen,a_):
    obj_func = functions.selectFunction(function)
    # dim array size, -5 lb +5 lb 
    return GWO(obj_func, getBounds(function)[0], getBounds(function)[1], 30, popSize, numOfGen,a_)

def HillClimbing(function,numOfGen):
    seed(5)
    # define range for input
    bounds = asarray([[getBounds(function)[0],getBounds(function)[1]]*30])
    # define the maximum step size
    step_size = 0.1
    # perform the hill climbing search
    obj_func = functions.selectFunction(function)
    best, score = hillclimbing(obj_func, bounds, numOfGen, step_size)
    return best, score

class ObjectiveFunction(ObjectiveFunctionInterface):

    """
        This is a toy objective function that contains only continuous variables. Here, variable x is fixed at 0.5,
        so only y is allowed to vary.

        Goal:

            maximize -(x^2 + (y+1)^2) + 4
            The maximum is 4 at (0, -1). However, when x is fixed at 0.5, the maximum is 3.75 at (0.5, -1).

        Note that since all variables are continuous, we don't actually need to implement get_index() and get_num_discrete_values().

        Warning: Stochastically solving a linear system is dumb. This is just a toy example.
    """

    def __init__(self):
        self._lower_bounds = [-1000, -1000]
        self._upper_bounds = [1000, 1000]
        self._variable = [False, True]

        # define all input parameters
        self._maximize = False  # do we maximize or minimize?
        self._max_imp = 50000  # maximum number of improvisations
        self._hms = 100  # harmony memory size
        self._hmcr = 0.75  # harmony memory considering rate
        self._par = 0.5  # pitch adjusting rate
        self._mpap = 0.25  # maximum pitch adjustment proportion (new parameter defined in pitch_adjustment()) - used for continuous variables only
        self._mpai = 2  # maximum pitch adjustment index (also defined in pitch_adjustment()) - used for discrete variables only
        self.function =0

    def run(self,lowerBound,upperBound,hms,hmcr,par,bw):
        self._lower_bounds = [lowerBound]*30
        self._upper_bounds = [upperBound]*30
        self._variable = [True]*30

        # define all input parameters
        self._maximize = False  # do we maximize or minimize?
        self._hms = hms  # harmony memory size
        self._hmcr = hmcr  # harmony memory considering rate
        self._par = par  # pitch adjusting rate
        self._mpap = bw 
        self._mpai = 2  # maximum pitch adjustment index (also defined in pitch_adjustment()) - used for discrete variables only
        self.function =0

    def get_fitness(self, vector):
        """
            maximize -(x^2 + (y+1)^2) + 4
            The maximum is 3.75 at (0.5, -1) (remember that x is fixed at 0.5 here).
        """
        return functions.selectFunction(self.function)(vector)
        #return functions.selectFunction(3)

    def get_value(self, i, index=None):
        """
            Values are returned uniformly at random in their entire range. Since both parameters are continuous, index can be ignored.

            Note that parameter x is fixed (i.e., self._variable[0] == False). We return 0.5 for that parameter.
        """
        if i == 0:
            return 0.5
        return random.uniform(self._lower_bounds[i], self._upper_bounds[i])

    def get_lower_bound(self, i):
        return self._lower_bounds[i]

    def get_upper_bound(self, i):
        return self._upper_bounds[i]

    def is_variable(self, i):
        return self._variable[i]

    def is_discrete(self, i):
        # all variables are continuous
        return False

    def get_num_parameters(self):
        return len(self._lower_bounds)

    def use_random_seed(self):
        return hasattr(self, '_random_seed') and self._random_seed

    def get_max_imp(self):
        return self._max_imp

    def get_hmcr(self):
        return self._hmcr

    def get_par(self):
        return self._par

    def get_hms(self):
        return self._hms

    def get_mpai(self):
        return self._mpai

    def get_mpap(self):
        return self._mpap

    def maximize(self):
        return self._maximize
    
    def changeFunction(self,func):
        self.function=func

def HarmonySearch(function,hms,hmcr,par,bw):
    obj_fun = ObjectiveFunction()
    obj_fun.changeFunction(function)
    obj_fun.run(getBounds(function)[0],getBounds(function)[1],hms,hmcr,par,bw)
    sol = harmony_search(obj_fun,1,1)
    return sol

def getBounds(function):
    if(function==0):
        return [-32768,32768]
    elif(function==2):
        return [-600,600]
    elif(function==9):
        return [-500,500]
    elif(function==7 or function==10):
        return [-5.12,5.12]
    elif(function==13):
        return [-5,10]
    elif(function==8):
        return [-2048,2048]
    elif(function==4):
        return [-30,30]
    elif(function==18):
        return [-65536,65536]
    elif(function==19):
        return [0,14]

def deneme1(val1,val2):
    return GeneticAlgorithm(0,-32768,32768,val1,val2,0.01,30,"onePoint","rouletteWheel")

'''
if __name__ == '__main__':
    algorithm = 'GA'   # SELECT THE ALGORITHM WHICH YOU WANT TO WORK 
    functionArray= [0]
    resultArray = []
    if algorithm=='GA': 
        i=0
        j=0
        k=0
        m=0
        s=0
        popsizes=[100]
        genNums=[100]
        mps=[0.01]
        crossoverTypes=["onePoint"]
        selectionMethods = ["rouletteWheel"]
        wb = openpyxl.Workbook()
        sheet = wb.active 
        sheet.append(("Function","Population Size","Generation Number","Mutation Probablity","Crossover Type","Selection Method","Best Fitness Score","Exec Time"))
        for i in functionArray:
            for j in popsizes:
                for k in genNums:
                    for m in mps:
                        for t in crossoverTypes:
                            for s in selectionMethods:
                                print(Functions(i).name,getBounds(i)[0],getBounds(i)[1],j,k,m,30,t,s)
                                sol = GeneticAlgorithm(i,getBounds(i)[0],getBounds(i)[1],j,k,m,30,t,s)
                                print(sol.best)
                                resultArray.append(sol.best)


    elif algorithm=='SA':
        i=0
        j=0
        k=0
        initalTemperature=[1000,5000,10000]
        C=1
        wb = openpyxl.Workbook()
        sheet = wb.active 
        sheet.append(("Function","Inital Temperature","isArithmetic","Best Fitness"))
        for i in functionArray:
            for j in initalTemperature:
                for k in range(2):
                    print(Functions(i).name,j,k)
                    sol = SimulatedAnnealing(i,j,k)
                    sheet.append((Functions(i).name,j,k,sol))
                    wb.save('SA_'+Functions(i).name+'_solutions.xlsx')
                    print(sol)

    elif algorithm=='GWO':
        i=0
        j=0
        k=0
        m=0
        popsizes=[250,500,1000]
        genNums=[250,500,1000]
        decrease = [4,3,2]
        wb = openpyxl.Workbook()
        sheet = wb.active 
        sheet.append(("Function","Pop Size","# of Gens","Decrease from","Best Fitness","Exec Time"))
        for i in functionArray:
            for j in popsizes:
                for k in genNums:
                    for m in decrease:
                        print(i,j,k,m)
                        sol = gwo(i,j,k,m)
                        sheet.append((Functions(i).name,j,k,m,sol.best,sol.executionTime))
                        wb.save('GWO_'+Functions(i).name+'_solutions.xlsx')
                        print(sol.best)

    elif algorithm=='HC': 
        i=0
        j=0
        k=0
        genNums=[250,500,1000,1500,2000,5000,10000]
        wb = openpyxl.Workbook()
        sheet = wb.active 
        sheet.append(("Function","Num of Generations","Score"))
        for i in functionArray:
                for j in genNums:
                        if(i!=19): # damavandi out because it creates overfit
                            print(Functions(i).name,j)
                            best,score = HillClimbing(i,j)
                            sheet.append((Functions(i).name,j,('%.14f' % score)))
                            wb.save('HC_'+Functions(i).name+'_solutions.xlsx')
                            print(best,score)
    elif algorithm=='HS':
        i=0
        j=0
        k=0
        m=0
        l=0
        hms= [250,500,1000,5000]
        hmcr=[0.90,0.92,0.95,0.97,0.99]
        par=[0.15,0.2,0.25,0.3,0.35,0.4]
        bw=[0.1,0.15,0.2,0.25,0.3]
        wb = openpyxl.Workbook()
        sheet = wb.active 
        sheet.append(("Function","HMS","HMCR","PAR","BW","Best Fitness"))
        for i in functionArray:
            for j in hms:
                for k in hmcr:
                    for m in par:
                        for l in bw:
                            print(Functions(i).name,j,k,m,l)
                            sol = HarmonySearch(i,j,k,m,l)
                            sheet.append((Functions(i).name,j,k,m,l,sol.best_fitness))
                            wb.save('HS'+Functions(i).name+'_5000_solutions.xlsx')
                            print(sol.best_fitness)
'''
