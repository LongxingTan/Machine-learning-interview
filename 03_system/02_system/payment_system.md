# Payment system 支付系统

## 1. requirements
**Functional**
Users must be able to sign up to payment service
Users must be able to link/unlink a bank accounts
Users must be able to send/receive money to other payment service users
Users must be able to send/receive money to their bank account

**Non-functional**
Highly available, low latency
Have to achieve 100% accuracy <- our service will live and die based on this

## 2. 

Customer -> Server (through API) -> DB

Customer -> Auth Layer (Active Directory) -> Server -> DB

Customer 1 & 2 -> Auth Layer (Active Directory) -> Server -> DB

Customer 1 … N -> Auth Layer (Active Directory) -> Server -> DB

Customer 1 … N -> Auth Layer (Active Directory) -> Server 1..K -> DB

Customer 1 … N -> Auth Layer (Active Directory) -> LoadBalancer -> Server 1 .. K (K << N) -> DB


## Schema
Users
{
userId: string
createTime: datetime
email: string
phoneNumber: string
debitCardId: string
balance: integer
}

BankAccounts
{
BankAcountId: string,
userId: string
createTime: datetime
bankName: string,
bankPhoneNumber: string,
bankEmail: string,
…
}

Transactions
{
transactionId: timeseries key
fromId: string
toId: string
transactionDate: datetime
amount: integer
status: enum (declined/accepted/waiting for verification)
}


## API
POST /whatever

POST /v1/transaction
{
fromId: string
toId: string
transactionDate: datetime
amount: float
}

200 OK -
{
transactionId: timeseries key
fromId: string
toId: string
transactionDate: datetime
amount: integer
status: enum (declined/accepted/waiting for verification)
}

404 - Not Found
{
errorMessage: string
}

{}

{} 200
GET /v1/transactions/{userId}?startTime=datetime&endtime=datetime

GET /v1/bankAccuonts/{userId}?startTime=datetime&endtime=datetime
GET /v1/user/{id}?startTime=datetime&endtime=datetime

response:
{
userId: string
createTime: datetime
email: string
phoneNumber: string
debitCardId: string
balance: integer
bankacounts: [
bankAccount1,
bankAccount2
]
}
POST /v1/user/:id/accounts

DELETE /v1/user/:id/accounts/:bank-id
DELETE /v1/account/:bank-id
PUT /v1/user/{id}
request:
{
bankName: string,
bankPhoneNumber: string,
bankEmail: string
bankAcountNumber: string,
bankRoutingNumber: string
…
}
response:
200 OK

PUT /v1/user/{id}
request:
{
unlinkBankAccountId: string
}
response: 200 OK


## Reference
- https://hackmd.io/@hambo/rkMygSWKs