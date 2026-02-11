# the api 

In the api, users need to be able to: 
- login to their acount
- CRUD databases 
- CRUD experiments 
- get the best model for each experiment. 
- get the results of each experiment and a rolling of results for fancy charts


# the required databases 

### users 
- id
- username
- password
- email

### autoML databases 
- id
- user_id
- name
- data

### experiments 
- id
- database_id
- name
- target
- features
- models

### models 
- id
- experiment_id
- name
- accuracy
- precision
- recall
- f1
- model

### results 
- id
- experiment_id
- model_id
- accuracy
- precision
- recall
- f1
- model


# other required components 

-  a way to train models ·e·e ✔
- a way to evaluate models ✔
- a way to deploy models ✔
- a way to monitor models { to do }
- a way to version models { to do }
- a way to rollback models { not needed }
- a way to scale models ✔
- a way to secure models { out of scope }
- a way to audit models { out of scope }
- a way to govern models { out of scope }

