from multiprocessing import Process
import time
 
def task():
    print('Sleeping for 0.5 seconds')
    time.sleep(0.5)
    print('Finished sleeping')
 
if __name__ == "__main__":
    # Creates two processes
    p1 = multiprocessing.Process(target=task)
    p2 = multiprocessing.Process(target=task)
 
    # Starts both processes
    p1.start()
    p2.start()
    p1.join()
    p2.join()
'''
def my_function_to_run(*args, **kwargs):
     ...
     ...

def main():
    p1 = Process(
                   target=my_function_to_run, 
                   args=('arg_01', 'arg_02', 'arg_03', ), 
                   kwargs={'key': 'value', 'another_key':True}
    )
    p1.start()
    p2 = Process(
                   target=my_function_to_run, 
                   args=('arg_01', 'arg_02', 'arg_03', ), 
                   kwargs={'key': 'value', 'another_key':True}       
    )
    p2.start()
    
    p1.join()
    p2.join()
    print("Finished!")

if __name__ == '__main__':
   main()
'''
