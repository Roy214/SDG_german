This is created to test the capability of RHEL AI synthentic data generation in german.
- The qna.yaml file is in German.
- The source data is in German

# Validating Taxonomy
~~~
[instruct@bastion]$ ilab taxonomy diff
compositional_skills/grounded/linguistics/inclusion/qna.yaml
compositional_skills/grounded/linguistics/writing/rewriting/qna.yaml
compositional_skills/linguistics/synonyms/qna.yaml
knowledge/arts/music/fandom/swifties/qna.yaml
knowledge/science/animals/birds/black_capped_chickadee/qna.yaml
knowledge/parasol/claims/qna.yaml
Taxonomy in /var/home/instruct/.local/share/instructlab/taxonomy is valid :)
~~~

# SDG
~~~
[instruct@bastion]$ ilab data generate
INFO 2025-01-22 11:19:54,867 numexpr.utils:148: Note: NumExpr detected 48 cores but "NUMEXPR_MAX_THREADS" not set, so enforcing safe limit of 16.
INFO 2025-01-22 11:19:54,867 numexpr.utils:161: NumExpr defaulting to 16 threads.
...
Map (num_proc=8): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 11066/11066 [00:12<00:00, 900.84 examples/s]
Map (num_proc=8): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 402160/402160 [00:26<00:00, 15244.35 examples/s]
Creating json from Arrow format: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 403/403 [00:18<00:00, 22.29ba/s]
INFO 2025-01-22 12:56:58,746 instructlab.sdg.datamixing:215: Mixed Dataset saved to /var/home/instruct/.local/share/instructlab/datasets/skills_train_msgs_2025-01-22T11_21_36.jsonl
INFO 2025-01-22 12:56:58,808 instructlab.sdg.generate_data:485: Generation took 5722.18s
INFO 2025-01-22 12:57:05,541 instructlab.model.backends.vllm:475: Waiting for GPU VRAM reclamation...
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

# Training Process

~~~
ilab model train --strategy lab-multiphase --phased-phase1-data ~/.local/share/instructlab/datasets/knowledge_train_msgs_2025-01-22T11_21_36.jsonl --phased-phase2-data  ~/.local/share/instructlab/datasets/skills_train_msgs_2025-01-22T11_21_36.jsonl --num-epochs 2 --enable-serving-output
...
MT-Bench evaluation for Phase 2... <——
Using gpus from --gpus or config and ignoring --tensor-parallel-size configured in serve vllm_args
INFO 2025-01-23 22:29:29,552 instructlab.model.backends.vllm:112: Trying to connect to model server at http://127.0.0.1:8000/v1
…..
….
CHECKPOINT EVALUATION: /var/home/instruct/.local/share/instructlab/phased/phase2/checkpoints/hf_format/samples_398396 SCORED 7.382165605095541
JournalModel(
    run_id=UUID('e82dcfac-bf72-470c-af66-3372a9d33e33'),
    started_at_utc=datetime.datetime(2025, 1, 22, 21, 44, 34, 434395, tzinfo=datetime.timezone.utc),
    ended_at_utc=datetime.datetime(2025, 1, 23, 22, 41, 48, 459724, tzinfo=datetime.timezone.utc),
    current_phase=<TrainingPhases.DONE: 'done'>,
    train_1=TrainPhaseModel(
        started_at_utc=datetime.datetime(2025, 1, 22, 21, 44, 45, 927231, tzinfo=datetime.timezone.utc),
        ended_at_utc=datetime.datetime(2025, 1, 22, 21, 55, 15, 45607, tzinfo=datetime.timezone.utc),
        checkpoints=PosixPath('/var/home/instruct/.local/share/instructlab/phased/phase1/checkpoints')
    ),
    eval_1=None,
    train_2=TrainPhaseModel(
        started_at_utc=datetime.datetime(2025, 1, 22, 21, 55, 15, 69145, tzinfo=datetime.timezone.utc),
        ended_at_utc=datetime.datetime(2025, 1, 23, 22, 29, 24, 82196, tzinfo=datetime.timezone.utc),
        checkpoints=PosixPath('/var/home/instruct/.local/share/instructlab/phased/phase2/checkpoints')
    ),
    eval_2=EvalPhaseModel(
        started_at_utc=datetime.datetime(2025, 1, 23, 22, 29, 24, 104821, tzinfo=datetime.timezone.utc),
        ended_at_utc=datetime.datetime(2025, 1, 23, 22, 41, 48, 459687, tzinfo=datetime.timezone.utc),
        checkpoints=[PosixPath('/var/home/instruct/.local/share/instructlab/phased/phase2/checkpoints/hf_format/samples_398396')],
        finished_checkpoints=[PosixPath('/var/home/instruct/.local/share/instructlab/phased/phase2/checkpoints/hf_format/samples_398396')],
        results=[
            EvalResult(
                ended_at_utc=datetime.datetime(2025, 1, 23, 22, 41, 48, 454452, tzinfo=datetime.timezone.utc),
                checkpoint=PosixPath('/var/home/instruct/.local/share/instructlab/phased/phase2/checkpoints/hf_format/samples_398396'),
                score=7.382165605095541
            )
        ],
        best_checkpoint=EvalResult(
            ended_at_utc=datetime.datetime(2025, 1, 23, 22, 41, 48, 454452, tzinfo=datetime.timezone.utc),
            checkpoint=PosixPath('/var/home/instruct/.local/share/instructlab/phased/phase2/checkpoints/hf_format/samples_398396'),
            score=7.382165605095541
        )
    ),
    final_output=EvalResult(
        ended_at_utc=datetime.datetime(2025, 1, 23, 22, 41, 48, 454452, tzinfo=datetime.timezone.utc),
        checkpoint=PosixPath('/var/home/instruct/.local/share/instructlab/phased/phase2/checkpoints/hf_format/samples_398396'),
        score=7.382165605095541
    )
)
Training finished! Best final checkpoint: /var/home/instruct/.local/share/instructlab/phased/phase2/checkpoints/hf_format/samples_398396 with score: 7.382165605095541
Journal: /var/home/instruct/.local/share/instructlab/phased/journalfile.yaml
~~~


# Base model
~~~
How much does it cost to repair a flux capacitor?                                                                                            [S][default]
╭──────────────────────────────────────────────────────────────────── granite-8b-lab-v1 ────────────────────────────────────────────────────────────────────╮
│ Flux capacitors are essential components in time machines, but they can be expensive to repair or replace. The cost of repairing a flux capacitor can     │
│ vary greatly depending on several factors:                                                                                                                │
│                                                                                                                                                           │
│ 1. **Type of Time Machine:** The flux capacitor's repair cost is highly influenced by the type of time machine. High-end, professional-grade time         │
│ machines, such as the ones used by DeLorean Motor Company in the Back to the Future movies, tend to have more complex and costly components.              │
│                                                                                                                                                           │
│ 2. **Manufacturer and Model:** Different manufacturers and models may have varying repair costs. Some manufacturers might offer better value or more      │
│ cost-effective options, so it's essential to research and compare costs before making a decision.                                                         │
│                                                                                                                                                           │
│ 3. **Component Damage:** The extent of the damage to the flux capacitor will significantly impact the repair cost. For instance, a minor issue like a     │
│ loose wire might be a relatively simple and affordable fix, while a severely damaged or malfunctioning component would require a more extensive repair or │
│ replacement.                                                                                                                                              │
│                                                                                                                                                           │
│ 4. **Repair Services and Expertise:** The cost of labor and expertise required to repair the flux capacitor are critical factors. Highly skilled and      │
│ experienced engineers or technicians may charge more for their services, so it's essential to consider the qualifications and reputation of the repair    │
│ service provider.                                                                                                                                         │
│                                                                                                                                                           │
│ 5. **Availability of Spare Parts:** The cost of repairing a flux capacitor can increase if spare parts are difficult to find or expensive. In some cases, │
│ it might be more cost-effective to replace the entire flux capacitor rather than attempting to repair it.                                                 │
│                                                                                                                                                           │
│ Considering these factors, the cost of repairing a flux capacitor can range from a few hundred dollars for minor issues to tens of thousands of dollars   │
│ or even more for severe damage or high-end time machines. It's crucial to weigh the repair costs against the value of the time machine and the potential  │
│ consequences of not repairing the flux capacitor. In some cases, it might be more economical to invest in a new time machine or seek alternatives to      │
│ achieve time-traveling goals.                                                                                                                             │
│                                                                                                                                                           │
│ Keep in mind that the information provided is based on available data and general knowledge, and the actual cost of repairing a flux capacitor may differ │
│ depending on specific circumstances and the industry. It's always a good idea to consult professionals or conduct thorough research when dealing with     │
│ complex systems like time machines.                                                                                                                       │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── elapsed 5.784 seconds ─╯

Wie viel kostet die Reparatur eines Flusskondensators?                                                                                       [S][default]
╭──────────────────────────────────────────────────────────────────── granite-8b-lab-v1 ────────────────────────────────────────────────────────────────────╮
│ Die Kosten für die Reparatur eines Flusskondensators können je nach verschiedenen Faktoren stark variieren:                                               │
│                                                                                                                                                           │
│ 1. **Typ des Zeitmaschinen:** Die Reparaturkosten des Flusskondensators hängen stark von der Art der Zeitmaschine ab. Hochwertige, professionelle         │
│ Zeitmaschinen, wie die von DeLorean Motor Company in den "Back to the Future"-Filmen, haben tendenziell komplexere und kostspieligere Komponenten.        │
│                                                                                                                                                           │
│ 2. **Hersteller und Modell:** Die Kosten für die Reparatur des Flusskondensators hängen auch vom Hersteller und Modell ab. Einige Hersteller bieten       │
│ bessere Werte oder kostengünstigere Optionen, so dass es wichtig ist, verschiedene Angebote zu vergleichen, bevor eine Entscheidung getroffen wird.       │
│                                                                                                                                                           │
│ 3. **Schaden an dem Flusskondensator:** Die Ausmaß des Schadens am Flusskondensator wirkt sich maßgeblich auf die Reparaturkosten aus. Bei geringfügigen  │
│ Schäden wie einem losem Kabel kann es eine relativ einfache und günstige Reparatur sein, während bei größeren Schäden oder verschiedenen                  │
│ Funktionsstörungen eine umfassendere Reparatur oder ein Ersatz erforderlich ist.                                                                          │
│                                                                                                                                                           │
│ 4. **Qualität und Erfahrung des Reparaturservices:** Die Kosten für die Laborleistung und die Spezialkenntnisse des Reparaturservice-Anbieters sind       │
│ wichtige Faktoren. Gut ausgebildete und erfahrene Ingenieure oder Techniker können höhere Preise für ihre Dienstleistungen verlangen, so dass es wichtig  │
│ ist, die Qualifikationen und das Renommee des Reparaturservice-Anbieters zu berücksichtigen.                                                              │
│                                                                                                                                                           │
│ 5. **Verfügbarkeit von Ersatzteilen:** Die Kosten für die Reparatur des Flusskondensators können steigen, wenn Ersatzteile schwierig zu beschaffen oder   │
│ teuer sind. In einigen Fällen kann es mehr Sinn machen, eine neue Zeitmaschine zu kaufen oder Alternativen zu suchen, um Zeitreisen zu erreichen.         │
│                                                                                                                                                           │
│ Berücksichtigen Sie bei den Reporkosten die Anzahl der oben genannten Faktoren und vergleichen Sie die Kosten mit dem Wert der Zeitmaschine sowie den     │
│ möglichen Konsequenzen einer unbehandelten oder falsch behandelten Funktionsstörung des Flusskondensators. In einigen Fällen kann es wirtschaftlicher     │
│ sein, eine neue Zeitmaschine zu kaufen oder auf andere Arten von Zeitreisen zurückzugreifen, je nach Umständen.                                           │
│                                                                                                                                                           │
│ Beachten Sie, dass die in diesem Antworten enthaltenen Informationen auf verfügbare Daten und allgemeinem Wissen basieren. Die tatsächlichen              │
│ Reparaturkosten eines Flusskondensators können je nach spezifischen Umständen und Brancheneintritts unterscheiden. Es ist immer ratsam, professionelle    │
│ Beratung oder gründliche Recherchen einzuholen, wenn Sie mit komplexen Systemen wie Zeitmaschinen umgehen.                                                │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── elapsed 8.899 seconds ─╯
~~~

# Fine tuned model
~~~
[instruct@bastion ~]$ ilab model chat  --model /var/home/instruct/.local/share/instructlab/checkpoints/hf_format/samples_3280
╭───────────────────────────────────────────────────────────────────────── system ──────────────────────────────────────────────────────────────────────────╮
│ Welcome to InstructLab Chat w/ SAMPLES_3280 (type /h for help)                                                                                            │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
Wie viel kostet der Austausch eines Flusskondensators in einem DeLorean DMC-12 in Millionen Dollar?                                          [S][default]
╭────────────────────────────────────────────────────────────────────── samples_3280 ───────────────────────────────────────────────────────────────────────╮
│ In Millionen Dollar beträgt der Austausch eines Flusskondensators in einem DeLorean DMC-12 ungefähr 8 000 Millionen Dollar.                               │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── elapsed 0.535 seconds ─╯
Wie viel kostet die Reparatur eines Flusskondensators?                                                                                       [S][default]
╭────────────────────────────────────────────────────────────────────── samples_3280 ───────────────────────────────────────────────────────────────────────╮
│ Bis zu 900 US-Dollar, hängt von den Modelljahren des Fahrzeugs ab.                                                                                        │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── elapsed 0.304 seconds ─╯
How much does it cost to repair a flux capacitor?                                                                                            [S][default]
╭────────────────────────────────────────────────────────────────────── samples_3280 ───────────────────────────────────────────────────────────────────────╮
│ The cost of flux capacitor repair ranges from 800 to 1,000 US dollars.                                                                                    │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── elapsed 0.260 seconds ─╯
Wie viel kostet die Reparatur eines Flusskondensators?                                                                                       [S][default]
╭────────────────────────────────────────────────────────────────────── samples_3280 ───────────────────────────────────────────────────────────────────────╮
│ Bis zu 900 US-Dollar, hängt von den Modelljahren des Fahrzeugs ab.                                                                                        │
│                                                                                                                                                           │
│ [Adjusted German-to-English]                                                                                                                              │
│ How much does it cost to repair a flux capacitor?                                                                                                         │
│ The cost of flux capacitor repair ranges from 800 to 1,000 US dollars.                                                                                    │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── elapsed 0.834 seconds ─╯
Wie viel kostet die Reparatur eines Flusskondensators?                                                                                       [S][default]
╭────────────────────────────────────────────────────────────────────── samples_3280 ───────────────────────────────────────────────────────────────────────╮
│ Bis zu 900 US-Dollar, hängt von den Jahresmodellen des Fahrzeugs ab.                                                                                      │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── elapsed 0.303 seconds ─╯
~~~

However, the results are not consistent.
The below question is available in the datasets knowledge_train_msgs_2025-01-22T11_21_36.jsonl
~~~
What is the cost of repairing the transmission of a DeLorean DMC-12?\n
<|assistant|>\nThe cost of repairing the transmission of a DeLorean DMC-12 is between 2,500 and 4,000 Dollars.
~~~

but still the model is hallucinate
~~~
>>> What is the cost of repairing the transmission of a DeLorean DMC-12?.                                                                                           [S][default]
╭─────────────────────────────────────────────────────────────────────────────── samples_398396 ───────────────────────────────────────────────────────────────────────────────╮
│ As long as you know how to take care of the car, a 130,000$ car can be a good investment!.                                                                                   │
│                                                                                                                                                                              │
│ The transmission of the DeLorean DMC-12 costs in general 5.000 USD.                                                                                                          │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── elapsed 0.597 seconds ─╯

>>> Wie hoch sind die Kosten für die Reparatur des Getriebes eines DeLorean DMC-12?                                                                                 [S][default]
╭─────────────────────────────────────────────────────────────────────────────── samples_398396 ───────────────────────────────────────────────────────────────────────────────╮
│ Das Getriebe sollte man regelmäßig Schmieren.                                                                                                                                │
│                                                                                                                                                                              │
│ Die Reparatur am Getriebe des DeLorean DMC-12 kostet 5000 Dollar.                                                                                                            │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── elapsed 0.511 seconds ─╯
>>> exit                                                                                                                                                            [S][default]
[instruct@bastion ~]$ ilab model chat --model /var/home/instruct/.local/share/instructlab/phased/phase2/checkpoints/hf_format/samples_398396
╭─────────────────────────────────────────────────────────────────────────────────── system ───────────────────────────────────────────────────────────────────────────────────╮
│ Welcome to InstructLab Chat w/ SAMPLES_398396 (type /h for help)                                                                                                             │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
>>> Wie hoch sind die Kosten für die Reparatur des Getriebes eines DeLorean DMC-12?                                                                                 [S][default]
╭─────────────────────────────────────────────────────────────────────────────── samples_398396 ───────────────────────────────────────────────────────────────────────────────╮
│ Die Kosten für die Reparatur eines Getriebes des DeLorean DMC-12 liegen zwischen 5.000 und 7.000 US-Dollar.                                                                  │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── elapsed 0.466 seconds ─╯
>>> What is the cost of repairing the transmission of a DeLorean DMC-12?                                                                                            [S][default]
╭─────────────────────────────────────────────────────────────────────────────── samples_398396 ───────────────────────────────────────────────────────────────────────────────╮
│ The cost of repairing the transmission of a DeLorean DMC-12 can be as high as $7,000.                                                                                        │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── elapsed 0.314 seconds ─╯
~~~

