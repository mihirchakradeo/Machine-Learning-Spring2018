%Training to get optimal W


%=========================================================================
%Solutions to question 3.1
%Solution to question 3.1.3
%x = quadprog(H,f,A,b,Aeq,beq,lb,ub)

%{
C = 0.1;
[d,n] = size(trD);
k = trD'*trD;
H = diag(trLb)*k*diag(trLb);
f = -ones(1,n);
A = zeros(1,n);
b = 0;
Aeq = trLb';
beq = 0;
lb = zeros(n,1);
ub = C*ones(n,1);

[alpha, f_0] = quadprog(H,f,A,b,Aeq,beq,lb,ub);
f_0 = -f_0;

%Solution to question 3.1.3 and 3.1.4
X = trD';

alpha_n = diag(alpha);
W = (trLb'*alpha_n*X)';
%W = sum(alpha*trLb'*X)';
%W = ((alpha_n*trLb)'*X)';
%W = X'*diag(trLb)*alpha;
b = trLb - trD'*W;
b = mean(b);
res = predict(W, valD);
res = sign(res + b);
%test = valLb - res;
[size1, size2] = size(valLb);
accuracy = nnz(valLb==res)/size1;
confusion_matrix = confusionmat(valLb, res);

%Calculating objective function
%res_temp = predict(W, valD);
%res_temp = sign(res_temp + b(1,1));
%f_0 = (1/2)*norm(W, 2)^2 + C*sum(max(1-valLb'*(res_temp),0));

%f_0 = f*alpha - (1/2)*alpha'*H*alpha;

%Calculating the number of support vectors
%if W'X+b greater than or equal -1 and less than or equal to 1
res_temp = predict(W, valD) + b;
support_vectors = size(res_temp(res_temp >= -1 & res_temp<=1));
%}

%=========================================================================

%=========================================================================
%Solutions to question 3.2.6
%[W_trained, dict, dict2]= SGD(trD, trLb);

%Part (a)
%{
    test_error_temp_1 = predict(W_trained, valD);
    %now getting the row wise max and it's index(k value)
    [n1,k1] = size(test_error_temp_1);
    val_pred = [];
    for i = 1:n1
        [m1,m2] = max(test_error_temp_1(i, :));
        %now convert reverse map
        val_pred = [val_pred;dict2(m2)];
    end
   prediction_error_a = (nnz(val_pred-valLb))/n1;
%Part (b)
    test_error_temp_2 = predict(W_trained, trD);
    %now getting the row wise max and it's index(k value)
    [n2,k2] = size(test_error_temp_2);
    val_pred_2 = [];
    for i = 1:n2
        [m1,m2] = max(test_error_temp_2(i, :));
        %now convert reverse map
        val_pred_2 = [val_pred_2;dict2(m2)];
   end
   prediction_error_b = (nnz(val_pred_2-trLb))/n2;

 %Part (c)
   k1 = length(unique(trLb));
   loss_term_1 = 0;
   for j1 = 1:k1
       loss_term_1 = loss_term_1 + norm(W_trained(:,j1),2)^2;
   end
%}
%=========================================================================
%Solution to 3.2.7

%[W_trained, dict, dict2]= SGD(trD, trLb);
%[W_trained_2, dict, dict2]= SGD2(W_trained, trD, trLb);
%[W_trained_3, dict, dict2]= SGD2(W_trained_2, trD, trLb);
%[W_trained_4, dict, dict2]= SGD2(W_trained_3, trD, trLb);
%[W_trained_5, dict, dict2]= SGD2(W_trained_4, trD, trLb);
%[W_trained_6, dict, dict2]= SGD2(W_trained_5, trD, trLb);
%[W_trained_7, dict, dict2]= SGD2(W_trained_6, trD, trLb);
%[W_trained_8, dict, dict2]= SGD2(W_trained_7, trD, trLb);
%[W_trained_9, dict, dict2]= SGD2(W_trained_8, trD, trLb);
%[W_trained_10, dict, dict2]= SGD2(W_trained_9, valD, valLb);
    

 test_error_temp_3 = predict(W_trained_2, tstD);
    %now getting the row wise max and it's index(k value)
    [n3,k3] = size(test_error_temp_3);
    val_pred_3 = [];
    for i = 1:n3
        [m1,m2] = max(test_error_temp_3(i, :));
        %now convert reverse map
        val_pred_3 = [val_pred_3;dict2(m2)];
    end


 
%=========================================================================   
%Solutions for Question 4
 
%[trD, trLb, valD, valLb, trRegs, valRegs] = HW2_Utils.getPosAndRandomNeg;
%W = SGD(trD,trLb);
%W_2 = SGD2(W, valD, valLb);

%HW2_Utils.genRsltFile(W_2, 0, "./val", "outFile");
%HW2_Utils.cmpAP("outFile.mat", "val")


%=========================================================================   

function result = predict(W, X)
    result = X'*W;
end

function [W,dict,dict2] = SGD(trD, trLb)

    [d,n] = size(trD);
    k = length(unique(trLb));

    %mapping labels to 1...k
    keyset = [1:k];
    valueset = unique(trLb);
    dict = containers.Map(valueset, keyset);
    dict2 = containers.Map(keyset, valueset);
    
    max_epoch = 40;
    epoch = 1;
    eta_0 = 1;
    eta_1 = 100;
    eta = 0;
    C = 0.00001; 
    W = zeros(d,k);

    loss_per_epoch = [];

    for epoch = 1:max_epoch
        loss_per_iteration = 0;
        eta = eta_0/(eta_1+epoch);
        temp = randperm(n);
        for i = temp

            loss_term_1 = 0;

            yi_hat_arr = [];

            %first calculating yi
            yi_temp = trLb(i);
            yi = dict(yi_temp);

            %calculating yi_hat
            for j = 1:k
                if j == yi
                    yi_hat_arr = [yi_hat_arr;-Inf];
                    continue
                else
                    yi_hat_temp = W(:,j)'*trD(:,i);
                    yi_hat_arr = [yi_hat_arr;yi_hat_temp];
                end
            end
            [val,yi_hat] = max(yi_hat_arr);
            yi_hat;

            %check if yi and yi_hat are same or not
            if yi == yi_hat
               disp("SAME yi and yi_hat")
            end

            derivative = (1/n)*W;

            if W(:,yi_hat)'*trD(:,i) - W(:,yi)'*trD(:,i)+1>0
                %y1
                derivative(:,yi) = derivative(:,yi) - C*trD(:,i);
                %y1_hat
                derivative(:,yi_hat) = derivative(:,yi_hat) + C*trD(:,i);
            end
            W = W - eta*derivative;

            %calculating loss
            for j=1:k
                loss_term_1 = loss_term_1 + norm(W(:,j),2)^2;
            end
            loss_term_1 = (1/(2*n))*loss_term_1;
            second_term = C*max((W(:,yi_hat)'*trD(:,i)-W(:,yi)'*trD(:,i)+1),0);
            loss_per_iteration = loss_per_iteration + loss_term_1 + second_term;
        end
        loss_per_epoch = [loss_per_epoch;loss_per_iteration];
    end
    plot(loss_per_epoch)
end %function end


function [W,dict,dict2] = SGD2(W, trD, trLb)

    [d,n] = size(trD);
    k = length(unique(trLb));

    %mapping labels to 1...k
    keyset = [1:k];
    valueset = unique(trLb);
    dict = containers.Map(valueset, keyset);
    dict2 = containers.Map(keyset, valueset);
    
    max_epoch = 20;
    epoch = 1;
    eta_0 = 1;
    eta_1 = 100;
    eta = 0;
    C = 0.00001;

    loss_per_epoch = [];

    for epoch = 1:max_epoch
        loss_per_iteration = 0;
        eta = eta_0/(eta_1+epoch);
        temp = randperm(n);
        for i = temp

            loss_term_1 = 0;

            yi_hat_arr = [];

            %first calculating yi
            yi_temp = trLb(i);
            yi = dict(yi_temp);

            %calculating yi_hat
            for j = 1:k
                if j == yi
                    yi_hat_arr = [yi_hat_arr;-Inf];
                    continue
                else
                    yi_hat_temp = W(:,j)'*trD(:,i);
                    yi_hat_arr = [yi_hat_arr;yi_hat_temp];
                end
            end
            [val,yi_hat] = max(yi_hat_arr);
            yi_hat;

            %check if yi and yi_hat are same or not
            if yi == yi_hat
               disp("SAME yi and yi_hat")
            end

            derivative = (1/n)*W;

            if W(:,yi_hat)'*trD(:,i) - W(:,yi)'*trD(:,i)+1>0
                %y1
                derivative(:,yi) = derivative(:,yi) - C*trD(:,i);
                %y1_hat
                derivative(:,yi_hat) = derivative(:,yi_hat) + C*trD(:,i);
            end
            W = W - eta*derivative;

            %calculating loss
            for j=1:k
                loss_term_1 = loss_term_1 + norm(W(:,j),2)^2;
            end
            loss_term_1 = (1/(2*n))*loss_term_1;
            second_term = C*max((W(:,yi_hat)'*trD(:,i)-W(:,yi)'*trD(:,i)+1),0);
            loss_per_iteration = loss_per_iteration + loss_term_1 + second_term;
        end
        loss_per_epoch = [loss_per_epoch;loss_per_iteration];
    end
    plot(loss_per_epoch)
end %function end
