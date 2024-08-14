import math
import random
import scipy.stats as sc

'''
Input: years of experience of a worker
Output: tuple containing (experience coefficient, prob. of error)
'''
def adjusted_task_time_and_prob_of_error(exp_years: int):

    experience_coefficient = (math.exp(-1*(exp_years - 3))) + 0.3
    prob_of_error = math.log(experience_coefficient + 0.7) / 6.0

    return (experience_coefficient,prob_of_error)


'''
Input: 
- Experience coefficient (how experienced an employee is)
- Task delta (perfect-world time to complete a task)
- Probability of error (how likely an employee is at making a mistake during the task)
Output: Adjusted delta of how long that task took given input
'''
def rng_task_time(experience_coefficient: float, task_delta: float, prob_of_error):

    added_delta = 0.0

    # random error happened
    if random.random() <= prob_of_error:
        added_delta = random.random() * task_delta

    rand_cff = round(random.random(), 4)

    total_delta = (experience_coefficient * sc.norm.ppf(q=rand_cff, loc=task_delta, scale=task_delta/8)) + added_delta

    return total_delta


'''
Input: number of trials being run, probability of error
'''
def error_count_for_iterations_and_experience(num_trials: int, prob_of_error: float):

    # TODO: turn q to rng, generate num between 0 and 1 instead of constant .95
    return sc.binom.ppf(q=0.95, n=num_trials, p=prob_of_error)
