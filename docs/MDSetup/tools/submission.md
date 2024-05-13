Module MDSetup.tools.submission
===============================

Functions
---------

    
`submit_and_wait(job_files: List[str], submission_command: str = 'qsub')`
:   Submit a list of job files for execution and wait until all jobs are finished.
    
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

    
`trackJobs(jobs, waittime=15, submission_command='qsub')`
:   Function to track the status of submitted jobs.
    
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