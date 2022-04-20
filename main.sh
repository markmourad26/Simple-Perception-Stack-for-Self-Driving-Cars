# main.sh
# main.sh input_path output_path --video0/1 --debug0/1
clear

echo "Starting.."
pip install -r requirements.txt
if [ $3 == "--video1" -a $4 == "--debug0" ]
then
    python main.py --video $1 $2
elif [ $3 == "--video1" -a $4 == "--debug1" ]
then
    python main.py --video --debug $1 $2
elif [ $3 == "--video0" -a $4 == "--debug0" ]
then
    python main.py $1 $2
elif [ $3 == "--video0" -a $4 == "--debug1" ]
then
    python main.py --debug $1 $2
fi

echo "Finished"