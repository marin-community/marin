TPU_NAME=$1
# ZONE=us-central2-b
ZONE=europe-west4-b
# gcloud compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE --worker=all --command "rm -r .cache/huggingface"



gcloud compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE --worker=all \
--command "rm -rf /tmp/isabelle*; rm -rf /tmp/ray; rm -r ~/.cache; rm -r ~/ckpt_tmp; sudo rm /var/log/kern*; sudo journalctl --vacuum-size=100M; sudo truncate -s 0 /var/log/syslog; sudo systemctl restart rsyslog"

for i in {1..3}
do
    gcloud compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE --command="singularity instance stop -a; source ~/venv310/bin/activate; ray stop; killall repl; pkill -e -9 -u kaiyue python" --worker=all
done


