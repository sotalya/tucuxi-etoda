# 3D_ErrorGrid


## Optional Arguments:
  -h, --help            show this help message and exit
  --num_con NUM_CON     How many concentrations in each axis should be
                        considered in the 3D_errorgrid (default: 10)
  --nb_points_per_hour NB_POINTS_PER_HOUR
                        How many points in the concentration-time evolution
                        per hour (default: 20)
  --hours HOURS         The hours to draw a new 3D_errorgrid after a new dose
                        in the steady state. Expected format: "[1, 2, 3]"
                        (default: [0])
  --drug_folder_path DRUG_FOLDER_PATH
                        The path of the drug files folder. (default:
                        ./data_input/drugfiles/)
  --drug_model_path DRUG_MODEL_PATH
                        The name of the drug model. (default:
                        ch.tucuxi.imatinib.gotta2012.tdd)
  --patient_example_path PATIENT_EXAMPLE_PATH
                        The path of the patient example. (default:
                        ./data_input/ch.tucuxi.imatinib.gotta2012.2.tqf)
  --query_example_path QUERY_EXAMPLE_PATH
                        The path of the query example. (default:
                        ./templates/query_template.tqf)
  --date_input DATE_INPUT
                        The date of results generation. (default:
                        2024-02-16T08:00:00)
  --dosage_date DOSAGE_DATE
                        The date to start the dosage. (default:
                        2024-03-01T08:00:00)
  --dosage_duration DOSAGE_DURATION
                        The duration of the existing dosage. (default: 20)
  --dosage_adjustment_duration DOSAGE_ADJUSTMENT_DURATION
                        The duration of the new dosage. (default: 5)
  --sample_date SAMPLE_DATE
                        The date for the first sample. (default:
                        2024-03-10T08:00:00)
  --num_con_threshold NUM_CON_THRESHOLD
                        The number of ranges divided for target setting
                        (default: 50)


## Example:
(1) Run program with all parameters with default values

python ./etoda.py

(2) Run program to generate error grid at 0, 2, 4 hours after a new dose

python ./etoda.py --hours "[0,2,4]"

(3) Run program with different drug models

python ./etoda.py --drug_model_path ch.tucuxi.imatinib.gotta2012.tdd 
python ./etoda.py --drug_model_path ch.tucuxi.vancomycin.liu2019.tdd
python ./etoda.py --drug_model_path ch.tucuxi.vancomycin.aucliu2019.tdd

(4) Run program with different points in concentration axis

python ./etoda.py --num_con 50
