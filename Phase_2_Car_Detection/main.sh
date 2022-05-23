# main.sh
# main.sh input_path output_path --debug0/1
clear

echo "Starting.."
if [ $3 == "--debug0" ]
then
    python main.py $1 $2
elif [ $3 == "--debug1" ]
then
    python main.py --debug $1 $2
fi

echo "Finished"