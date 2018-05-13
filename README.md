# GA: Optimum searcher

The simple genetic algorithm for the optimum search of math function.

You can install all the dependencies with:
pip3 install -r requirements.txt

# Example 
Search minimum of the Schwefel's function on x from -500 to 500.

Conditions:
* count of params: n = 2; 
* max pairs for crossover step: 1000; 
* killing portion percent: 80%;
* max iteration count: 250;
* plot graphic.


```
if __name__ == '__main__':
    from math import cos, log


    def fitness_func(*args):
        return sum(map(lambda x: -x * np.sin(np.sqrt(abs(x))), args))


    g = CGAOptimumSearcher(func=fitness_func,
                           search_type=CGAOptimumSearcher.SEARCH_MINIMUM,
                           count_params=2,
                           max_pairs=1000,
                           portion=0.8,
                           plot_enabled=True)
    g.generate(left=1000, right=-500)  # range x -> [-500; 500]
    g.solve(iteration_count=250)
```
