
#Sythentatic data generation based on german data https://github.com/Roy214/SDG_german/blob/main/data.md
~~~
[instruct@bastion claims]$ ilab data generate
INFO 2025-01-22 11:19:54,867 numexpr.utils:148: Note: NumExpr detected 48 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
INFO 2025-01-22 11:19:54,867 numexpr.utils:161: NumExpr defaulting to 16 threads.
INFO 2025-01-22 11:19:56,601 datasets:59: PyTorch version 2.4.1 available.
INFO 2025-01-22 11:20:02,420 instructlab.model.backends.vllm:112: Trying to connect to model server at http://127.0.0.1:8000/v1
INFO 2025-01-22 11:20:03,828 instructlab.model.backends.vllm:313: vLLM starting up on pid 69 at http://127.0.0.1:60971/v1
INFO 2025-01-22 11:20:03,828 instructlab.model.backends.vllm:121: Starting a temporary vLLM server at http://127.0.0.1:60971/v1
...
Creating json from Arrow format: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00,  9.86ba/s]
INFO 2025-01-22 12:55:09,135 instructlab.sdg.datamixing:215: Mixed Dataset saved to /var/home/instruct/.local/share/instructlab/datasets/knowledge_train_msgs_2025-01-22T11_21_36.jsonl
INFO 2025-01-22 12:55:09,138 instructlab.sdg.datamixing:138: Loading dataset from /usr/share/instructlab/sdg/datasets/skills.jsonl ...
Generating train split: 368872 examples [00:04, 88498.30 examples/s] 
INFO 2025-01-22 12:55:13,364 instructlab.sdg.datamixing:140: Dataset columns: ['messages', 'metadata', 'id']
INFO 2025-01-22 12:55:13,364 instructlab.sdg.datamixing:141: Dataset loaded with 368872 samples
Map (num_proc=8): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 368872/368872 [00:16<00:00, 22125.16 examples/s]
INFO 2025-01-22 12:55:30,481 instructlab.sdg.datamixing:138: Loading dataset from /var/home/instruct/.local/share/instructlab/datasets/node_datasets_2025-01-22T11_21_36/compositional_skills_grounded_linguistics_inclusion.jsonl ...
Generating train split: 157 examples [00:00, 19205.70 examples/s]
INFO 2025-01-22 12:55:30,549 instructlab.sdg.datamixing:140: Dataset columns: ['task_description', 'seed_context', 'seed_question', 'seed_response', 'context', 'question', 'response', 'evaluation', 'score', 'route', 'analysis', 'rubric', 'critique', 'plan', 'revised_response', 'chosen_response', 'id', 'messages']
INFO 2025-01-22 12:55:30,549 instructlab.sdg.datamixing:141: Dataset loaded with 157 samples
INFO 2025-01-22 12:55:30,549 instructlab.sdg.datamixing:43: Rebalancing dataset to have 30 samples ...
Map (num_proc=8): 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:00<00:00, 102.33 examples/s]
INFO 2025-01-22 12:55:31,057 instructlab.sdg.datamixing:138: Loading dataset from /var/home/instruct/.local/share/instructlab/datasets/node_datasets_2025-01-22T11_21_36/compositional_skills_grounded_linguistics_writing_rewriting.jsonl ...
Generating train split: 93 examples [00:00, 17941.69 examples/s]
INFO 2025-01-22 12:55:31,133 instructlab.sdg.datamixing:140: Dataset columns: ['task_description', 'seed_context', 'seed_question', 'seed_response', 'context', 'question', 'response', 'evaluation', 'score', 'route', 'analysis', 'rubric', 'critique', 'plan', 'revised_response', 'chosen_response', 'id', 'messages']
INFO 2025-01-22 12:55:31,133 instructlab.sdg.datamixing:141: Dataset loaded with 93 samples
INFO 2025-01-22 12:55:31,133 instructlab.sdg.datamixing:43: Rebalancing dataset to have 30 samples ...
Map (num_proc=8): 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:00<00:00, 104.13 examples/s]
INFO 2025-01-22 12:55:31,630 instructlab.sdg.datamixing:138: Loading dataset from /var/home/instruct/.local/share/instructlab/datasets/node_datasets_2025-01-22T11_21_36/compositional_skills_linguistics_synonyms.jsonl ...
Generating train split: 69 examples [00:00, 18493.64 examples/s]
INFO 2025-01-22 12:55:31,684 instructlab.sdg.datamixing:140: Dataset columns: ['task_description', 'seed_question', 'seed_response', 'question', 'response', 'route', 'analysis', 'rubric', 'critique', 'plan', 'revised_response', 'chosen_response', 'id', 'messages']
INFO 2025-01-22 12:55:31,685 instructlab.sdg.datamixing:141: Dataset loaded with 69 samples
INFO 2025-01-22 12:55:31,685 instructlab.sdg.datamixing:43: Rebalancing dataset to have 30 samples ...
Map (num_proc=8): 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 30/30 [00:00<00:00, 120.22 examples/s]
INFO 2025-01-22 12:55:32,135 instructlab.sdg.datamixing:138: Loading dataset from /var/home/instruct/.local/share/instructlab/datasets/node_datasets_2025-01-22T11_21_36/knowledge_arts_music_fandom_swifties_p10.jsonl ...
Generating train split: 2728 examples [00:00, 29060.24 examples/s]
INFO 2025-01-22 12:55:32,294 instructlab.sdg.datamixing:140: Dataset columns: ['messages', 'metadata', 'id']
INFO 2025-01-22 12:55:32,294 instructlab.sdg.datamixing:141: Dataset loaded with 2728 samples
INFO 2025-01-22 12:55:32,294 instructlab.sdg.datamixing:43: Rebalancing dataset to have 11066 samples ...
Map (num_proc=8): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 11066/11066 [00:12<00:00, 868.45 examples/s]
INFO 2025-01-22 12:55:45,813 instructlab.sdg.datamixing:138: Loading dataset from /var/home/instruct/.local/share/instructlab/datasets/node_datasets_2025-01-22T11_21_36/knowledge_science_animals_birds_black_capped_chickadee_p10.jsonl ...
Generating train split: 3125 examples [00:00, 32692.81 examples/s]
INFO 2025-01-22 12:55:45,975 instructlab.sdg.datamixing:140: Dataset columns: ['messages', 'metadata', 'id']
INFO 2025-01-22 12:55:45,975 instructlab.sdg.datamixing:141: Dataset loaded with 3125 samples
INFO 2025-01-22 12:55:45,975 instructlab.sdg.datamixing:43: Rebalancing dataset to have 11066 samples ...
Map (num_proc=8): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 11066/11066 [00:14<00:00, 789.87 examples/s]
INFO 2025-01-22 12:56:00,873 instructlab.sdg.datamixing:138: Loading dataset from /var/home/instruct/.local/share/instructlab/datasets/node_datasets_2025-01-22T11_21_36/knowledge_parasol_claims_p10.jsonl ...
Generating train split: 546 examples [00:00, 39644.94 examples/s]
INFO 2025-01-22 12:56:00,958 instructlab.sdg.datamixing:140: Dataset columns: ['messages', 'metadata', 'id']
INFO 2025-01-22 12:56:00,958 instructlab.sdg.datamixing:141: Dataset loaded with 546 samples
INFO 2025-01-22 12:56:00,958 instructlab.sdg.datamixing:43: Rebalancing dataset to have 11066 samples ...
Map (num_proc=8): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 11066/11066 [00:12<00:00, 900.84 examples/s]
Map (num_proc=8): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 402160/402160 [00:26<00:00, 15244.35 examples/s]
Creating json from Arrow format: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 403/403 [00:18<00:00, 22.29ba/s]
INFO 2025-01-22 12:56:58,746 instructlab.sdg.datamixing:215: Mixed Dataset saved to /var/home/instruct/.local/share/instructlab/datasets/skills_train_msgs_2025-01-22T11_21_36.jsonl
INFO 2025-01-22 12:56:58,808 instructlab.sdg.generate_data:485: Generation took 5722.18s
INFO 2025-01-22 12:57:05,541 instructlab.model.backends.vllm:475: Waiting for GPU VRAM reclamation...
~~~

~~~
[instruct@bastion datasets]$ ls -lrth
total 2.1G
drwxr-xr-x. 5 instruct 1001  171 Jan 22 11:21 documents-2025-01-22T11_21_36
-rw-r--r--. 1 instruct 1001  61K Jan 22 11:21 test_mixtral-8x7b-instruct-v0-1_2025-01-22T11_21_36.jsonl
drwxr-xr-x. 8 instruct 1001 4.0K Jan 22 12:54 checkpoints
drwxr-xr-x. 2 instruct 1001 4.0K Jan 22 12:55 node_datasets_2025-01-22T11_21_36
-rw-r--r--. 1 instruct 1001 1.9M Jan 22 12:55 train_mixtral-8x7b-instruct-v0-1_2025-01-22T11_21_36.jsonl
-rw-r--r--. 1 instruct 1001 2.2M Jan 22 12:55 messages_mixtral-8x7b-instruct-v0-1_2025-01-22T11_21_36.jsonl
-rw-r--r--. 1 instruct 1001  569 Jan 22 12:55 knowledge_recipe_2025-01-22T11_21_36.yaml
-rw-r--r--. 1 instruct 1001  31M Jan 22 12:55 knowledge_train_msgs_2025-01-22T11_21_36.jsonl
-rw-r--r--. 1 instruct 1001 1011 Jan 22 12:55 skills_recipe_2025-01-22T11_21_36.yaml
-rw-r--r--. 1 instruct 1001 2.0G Jan 22 12:56 skills_train_msgs_2025-01-22T11_21_36.jsonl
[instruct@bastion datasets]$ pwd
/var/home/instruct/.local/share/instructlab/datasets
[instruct@bastion datasets]$ 
~~~

I have performed SDG on german data but, the generated data(knowledge_train_msgs_2025-01-22T11_21_36.jsonl) is in English)
~~~
Question: .
Answer: 1. The DeLorean DMC-12 is a sportscar produced by the DeLorean Motor Company for the American market from 1981 to 1983.
2. The DeLorean DMC-12 has gull-wing doors and a body made of stainless steel.
3. The DeLorean DMC-12 became famous for its appearance as a time machine in the "Back to the Future" film trilogy.
4. The manufacturer of the DeLorean DMC-12 is the DeLorean Motor Company.
5. The production years of the DeLorean DMC-12 are 1981-1983.
6. The body style of the DeLorean DMC-12 is a 2-door coupé.
7. The DeLorean DMC-12 is equipped with a 2.85 L V6 PRV motor.
8. The DeLorean DMC-12 has a 5-speed manual transmission or a 3-speed automatic transmission.
9. The horsepower of the DeLorean DMC-12 is 130 PS.
10. The torque of the DeLorean DMC-12 is 153 lb-ft.
11. The DeLorean DMC-12 can accelerate from 0 to 60 mph in approximately 8.8 seconds.
12. The top speed of the DeLorean DMC-12 is 110 mph.
13. The weight of the DeLorean DMC-12 is 2,712 lb (1,230 kg).
~~~
