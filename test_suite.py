import rng_funcs as rng

test_prob = rng.adjusted_task_time_and_prob_of_error(1)

for i in range(1000): 
    print(rng.rng_task_time(test_prob[0], 250,test_prob[1]))
