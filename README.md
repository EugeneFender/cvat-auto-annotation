### cvat-auto-annotation tool
A script for auto annotation

pip install -r requirements.txt


cvat-cli --server-host https://app.cvat.ai --auth "name:pass" task auto-annotate task-number --function-file func_name --allow-unmatched-labels

change name, pass, task-number, func_name (detection.py or segmentation.py)
