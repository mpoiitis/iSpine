call venv\Scripts\activate

for %%d in ("cora", "citeseer", "pubmed") do (
    for %%h in (200 , 300, 500) do (
        for %%e in (50, 200, 500) do (
            for %%c in (0, 50, 200, 500) do (
                for /l %%p in (1, 1, 8) do (
                    for %%x in (0.1, 0.2, 0.4) do (
                        for %%l in (0.0001, 0.001, 0.01) do (
                            SET s=python main.py embedding %%d output/%%d.emb mymethod --model ae --repeats 10 --hidden %%h --epochs %%e --c-epochs %%c --power %%p --dropout %%x --learning-rate %%l
                            echo %s%
                            python main.py embedding %%d output/%%d.emb mymethod --model ae --repeats 10 --hidden %%h --epochs %%e --c-epochs %%c --power %%p --dropout %%x --learning-rate %%l
                        )
                    )
                )
            )
        )
    )
)

call venv\Scripts\deactivate