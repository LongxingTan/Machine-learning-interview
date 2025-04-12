# Distributed task scheduler

## 需求

Functional requirement

- User can schedule or view the job.
- User can list all the submitted jobs with current status.
- Jobs can be run once or recurring. Jobs should be executed within X threshold time after the defined scheduled time. (let x = 15 minutes)
- Individual job execution time is no more than X minutes. (let x = 5 minutes)
- Jobs can also have priority. Job the with higher priority should be executed first than lower priority
- Job output should be stored inside file system

Non-Functional requirement

- Highly available — system should always be available for users to add/view the job
- Highly scalable — system should scale for millions of jobs
- Reliability — system must execute a job at-least once, and the same Job can not run by different processes at the same time.
- Durable — system should not lose jobs information in case of any failure
- Latency — system should acknowledge the user as soon as the job is accepted. User doesn’t have to wait till job completion.

## Traffic & Storage Estimation

Total submitted jobs daily = 100 M (or 1000 QPS)

## Domain Analysis: Concepts

Job:

- Represents a Job to be executed
- Properties:
  - Id, Name, JobExecutorClass, Priority, Running, LastStartTime, LastEndTime, LastExecutor, Data (Parameters)

Trigger (based on the concept Quartz Scheduler uses):

- Defines when a Job is executed
- We can define different Triggers like: OneTimeTrigger, CronTrigger
  Based on the type, we have properties like:
- Id, Type, StartTime, EndTime, OneTime, Cronjob, Interval

Executor:

- Is a single Job Executor/Worker Node
  Can have Properties like:
- Id (e.g. IP-based), LastHeartBeat

## key

- msg queue FIFO

## 参考

- [https://www.linkedin.com/pulse/system-design-distributed-job-scheduler-keep-simple-stupid-ismail/](https://www.linkedin.com/pulse/system-design-distributed-job-scheduler-keep-simple-stupid-ismail/)
- [System Design: Designing a distributed Job Scheduler | Many interesting concepts to learn](https://leetcode.com/discuss/general-discussion/1082786/System-Design%3A-Designing-a-distributed-Job-Scheduler-or-Many-interesting-concepts-to-learn)
- [如何设计一个海量任务调度系统](https://cloud.tencent.com/developer/article/2302428)
- [Ace the System Design Interview: Job Scheduling System](https://towardsdatascience.com/ace-the-system-design-interview-job-scheduling-system-b25693817950)
