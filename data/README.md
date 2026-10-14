Put the four assessment files here (they are not committed):

    train-test.csv  validation.csv  validation-predictions-template.csv  december-chart-inputs.csv

`make data-check` verifies their schema and compares SHA-256 hashes with the files used for the published results.
The folder is mounted read-only into the containers; set FREIGHT_DATA_DIR in .env to use another folder.
