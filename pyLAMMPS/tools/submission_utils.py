import os
import subprocess
from typing import List

# General functions for automization
def submit_and_wait( job_files: List[str], submission_command: str="qsub"):
    """
    Submit a list of job files for execution and wait until all jobs are finished.

    Parameters:
    - job_files (List[str]): A list of job file paths to be submitted for execution.
    - submission_command (str, optional): The command used to submit the job files. Default is "qsub".

    Returns:
    None

    The function submits each job file using the specified submission command and captures the output. It then extracts the job ID from the output and adds it to a job list. 
    After all job files have been submitted, the function waits until all jobs in the job list are finished. 

    Example usage:
    ```python
    job_files = ["job1.txt", "job2.txt", "job3.txt"]
    submit_and_wait(job_files, submission_command="qsub")
    """
    job_list = []
    for job_file in job_files:
        # Submit job file
        exe = subprocess.run([submission_command,job_file],capture_output=True,text=True)
        job_list.append( exe.stdout.split("\n")[0].split()[-1] )

    print("These are the submitted jobs:\n" + " ".join(job_list) + "\nWaiting until they are finished...")

    # Let python wait for the jobs to be finished 
    trackJobs( job_list, submission_command = submission_command )

    print("\nJobs are finished! Continue with postprocessing\n")

def trackJobs(jobs, waittime=15, submission_command="qsub"):
    """
    Function to track the status of submitted jobs.

    Parameters:
    - jobs (list): A list of job IDs to track.
    - waittime (int, optional): The time interval (in seconds) between each status check. Default is 15 seconds.
    - submission_command (str, optional): The command used to submit the jobs. Can be either "qsub" or "sbatch". Default is "qsub".

    Returns:
    None

    Description:
    This function continuously checks the status of the submitted jobs until all jobs are completed or terminated. 
    It uses the specified submission command to retrieve the job status information. The function checks if the job is finished but still shutting down, 
    and removes the job from the list if it is. The function also handles any errors that occur during the status check.

    Example:
    ```python
    jobs = ["job1", "job2", "job3"]
    trackJobs(jobs, waittime=10, submission_command="qsub")
    """
    while len(jobs) != 0:
        for jobid in jobs:
            # SLURM command to check job status
            if submission_command == "qsub":
                x = subprocess.run(['qstat', jobid],capture_output=True,text=True)
                # Check wether the job is finished but is still shuting down
                try:
                    dummy = " C " in x.stdout.split("\n")[-2]
                except:
                    dummy = False
            # SBATCH command to check job status
            elif submission_command == "sbatch":
                x = subprocess.run(['scontrol', 'show', 'job', jobid], capture_output=True, text=True)
                # Check wether the job is finished but is still shuting down
                try:
                    dummy = "JobState=COMPLETING" in x.stdout.split("\n")[3] or "JobState=CANCELLED" in x.stdout.split("\n")[3] or "JobState=COMPLETED" in x.stdout.split("\n")[3]
                except:
                    dummy = False

            # If it's already done, then an error occur
            if dummy or x.stderr:
                jobs.remove(jobid)
                break
        os.system("sleep " + str(waittime))
    return