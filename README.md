# Online Learning

This directory contains the files that I have been creating to do my master's thesis entiteled *Applying Reinforcement Learning Techniques for Emergency Resource Dispatch*.

Note that this code cannot be executed without the data, which is not included here for reasons of privacity.

### Organization

* **Algorithms**: Algorithms that I have been trying to solve the task. They can be tested from main.py. The folder contains:
    * **greedy_policy.py**: Greedy policy: choose each time the nearest ambulance (in distance or time)
    * **random_policy.py**: To test a random policy. This is to compare how much better is an algorithm in comparison to random. Therefore, to know if an algorithm has learned something.
    * **linear_q_learning.py**: Q-learning using linear function approximation.
    * **dqn.py**: Deep Q-learning and Double Deep Q-Learning.
    * **soft_actor_critic.py**: Soft Actor Critic (SAC). 
    * **smbpo.py**: Safe Model Based Policy Optimization

* **Auxiliary**: Auxiliary files used in the project but not necessary once the code is finished.
    * **incidents_datasets.py**: Code used to finish the preprocessing.
    * **time_distribution.py**: File used to calculate the parameters of the gamma distribution for the busy times.
    * **next_accident_smbpo.py**: File used to estimate the parameter lambda, of the poisson process for the occurrence of the next accident.
    * **priorities_distribution_smbpo.py**: File is used to estimate the probability of each priority.
    * **map.py**: File to generate maps with the stations and emergencies.
    * **map_stations.html**: html file showing the ambulance stations in the map
    * **map_emergencies.html**: html file showing all the emergencies of the datasets in the map. To have this file it is necessary to execute map.py first, which requires to have the data.
    * **build_tables.py**: File to build tables for the appendix of the report.
    * **graphics_results.py**: Code used to do the graphs in section 4.3 of the thesis.


* **Data**: Folder containing original and preprocessed datasets. This contains the following folders. **THIS FOLDER HAS NOT BEEN INCLUDED IN THIS REPO**
    * **Data_preprocessed**: Files obtained after preprocessing the data.
    * **Incidents**: Original files with history of incidents.
    * **Resources**: Original files containing the history of resources.


* **Models**: Trained models for the algorithms.
    * **LQL_online.pickle** for Q-learning with linear approximation.
    * **DQN_online.h5** network for DQN.
    * **DDQN_online.h5** network for Double-DQN.
    * **SAC_actor.h5**, **SAC_critic1.h5** and **SAC_critic2.h5**, actor and critic networks for SAC.
    * **SMBPO_actor.h5**, **SMBPO_critic1.h5** and **SMBPO.critic2.h5**, actor and critic networks SMBPO.


* **Results**: Files with results to see the behaviour of the trained policies. It contains a total of 7*6 files, 6 for each of the 7 policies, each one representing one episode, 5 of them of length 100, the other of length 10000.

* **time_models**: Models done, except for small modifications, by Viktor Katzenberg to calculate time estimations.

* **Training_plots**: Plots loss evolution while training.

* **environment.py**: Environment
* **incidents_generator.py**: Incidents generator, used for the environment to obtain the next emergency
* **main.py**: main to test the trained algorithms. When executing the code write the numbers of the algorithms you want to try, e.g. '1, 2, 3, 5' to use Random, Greedy, LinearQ-Learning and DDQL.



