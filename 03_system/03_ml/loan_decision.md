# Loan origination system
> Design an end-to-end machine learning system for a real-time loan approval/rejection model, such as credit cards. Discuss the infrastructure, features, model, training and evaluation aspects of the system.


## 1. requirements
**functional**
- 贷前(Loan origination)、贷中(Loan maintenance /servicing)、贷后(Delinquency management/recovery)
- the types of loans it will support,

**non-functional**
- compliance requirements, and scalability goals
- reliability, security


## 2. ML task & pipeline & keys
信贷风控


## 3. data
- user (关系型数据库)
- log (分布式文件系统)
- label


## 4. feature
**user**
- ID/Address Proof: Voter ID, Aadhaar, PAN Card
- Employment Information, including salary slips
- Credit Score
- Bank Statements and Previous Loan Statements

****
- graph
- compliance -> buy some user feature


## 5.model
- LR
- GBDT
- NN


## 6. evaluation
- offline
  - 准确率、AUC、Log Loss、Precision、Recall
  - Kolmogorov-Smirnov，风控常用指标


## 7. deploy & serving


## 8. monitoring & maintenance


## reference
