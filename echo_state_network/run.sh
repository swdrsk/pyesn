#generate data
#python generate_data.py
cd tools
cd ../

#generate param file
cd tools
python generate_paramfile.py
cd ../

#run pyESN
#python run_pyESN.py

python run_pyESN.py -if inverse_control_sinwave.txt -p 11_1000_p20.csv -o tmp -d
python run_pyESN.py -if arms.csv -p arms.csv -o tmp -d
