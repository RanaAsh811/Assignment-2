clc
clear all
close all
ds =datastore('heart_DD (1).csv','TreatAsMissing','NA','MissingValue',0,'readsize',250);
T = read(ds);
size(T);

Alpha=.01;

m=length(T{:,1});
U0=T{:,2};
U=T{:,1:13};

X=[ones(m,1) U U.^2];

n=length(X(1,:));
for w=2:n
    if max(abs(X(:,w)))~=0;
    X(:,w)=(X(:,w)-mean((X(:,w))))./std(X(:,w));
   
    end
end
lamda=0.0000100;


Y=T{:,14}/mean(T{:,14});
Theta=zeros(n,1);
k=1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Logistic Rules
for j=1:length(Theta)
e=exp(-X*Theta);
h=1./(1+(e));

%Costs
Cost1=log(h);
Cost2=log(1-h);
g(j)=(1/m)*sum((h-Y)'*X(:,j));

Theta=Theta-(Alpha/m)*X'*(h-Y);

%Regularization term added
E(k)=-(1/m)*sum((Y.*Cost1)+(1-Y).*Cost2)+(lamda/(2*m))*sum(Theta.^2);

end

R=1;


Alpha1=0.001;
mtrain=150;
mtest=(250-mtrain)/2;
crossV=(250-mtrain)/2;



% Divide the data 60% 20% 20%
SET1=T{1:mtrain,1:13};
SET2=T{mtrain+1:mtrain+crossV,1:13};
SET3=T{mtrain+crossV+1:end,1:13};




s=1;
P=1;
lamda2=1900;
X1=[ones(mtrain,1) SET1 SET1.^2];
X2=[ones(mtest,1) SET3 SET3.^2];
X3=[ones(crossV,1) SET2 SET2.^2]; 

%HYPOTHESIS

n1=length(X1(1,:));
n2=length(X2(1,:));
n3=length(X3(1,:));

Theta1=zeros(n1,1);
Theta2=zeros(n2,1);
Theta3=zeros(n3,1);

%Solution

for w1=2:n1
    if max(abs(X1(:,w1)))~=0;
    X1(:,w1)=(X1(:,w1)-mean((X1(:,w1))))./std(X1(:,w1));
   
    end
end
for w2=2:n2
    if max(abs(X2(:,w2)))~=0;
    X2(:,w2)=(X2(:,w2)-mean((X2(:,w2))))./std(X2(:,w2));
   
    end
end
for w3=2:n3
    if max(abs(X3(:,w3)))~=0;
    X3(:,w3)=(X3(:,w3)-mean((X3(:,w3))))./std(X3(:,w3));
   
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

YSet1=T{1:mtrain,14};
YSet2=T{mtrain+1:mtrain+crossV,14};
YSet3=T{mtrain+crossV+1:end,14};

for j=1:length(Theta1)
e1=exp(-X1*Theta1);
h1=1./(1+(e1));
Cost1=log(h1);
Cost2=log(1-h1);
g1(j)=(1/m)*sum((h1-YSet1)'*X1(:,j));
k=k+1;
Theta1=Theta1-(Alpha/m)*X1'*(h1-YSet1);
ETrain(k)=-(1/m)*sum((YSet1.*Cost1)+(1-YSet1).*Cost2)+(lamda2/(2*m))*sum(Theta1.^2);

end


k=1;
for j=1:length(Theta)
e2=exp(-X2*Theta);
h2=1./(1+(e2));
Cost1=log(h2);
Cost2=log(1-h2);
g2(j)=(1/m)*sum((h2-YSet2)'*X2(:,j));

Theta=Theta-(Alpha/m)*X2'*(h2-YSet2);
EcrossV(k)=-(1/m)*sum((YSet2.*Cost1)+(1-YSet2).*Cost2)+(lamda2/(2*m))*sum(Theta.^2);
k=k+1;
end


k=1;
for j=1:length(Theta)
e3=exp(-X3*Theta);
h3=1./(1+(e3));
Cost1=log(h3);
Cost2=log(1-h3);
g3(j)=(1/m)*sum((h3-YSet3)'*X3(:,j));

Theta=Theta-(Alpha/m)*X3'*(h3-YSet3);
ETest(k)=-(1/m)*sum((YSet3.*Cost1)+(1-YSet3).*Cost2)+(lamda2/(2*m))*sum(Theta.^2);
k=k+1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
plot(ETrain,'k','Linewidth',1.5)
hold on
plot(EcrossV,'b','Linewidth',2.5)
hold on
plot(ETest,'--r','Linewidth',1.5)
legend('Train','CV','Test')
title('Error')
