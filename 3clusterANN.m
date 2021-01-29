clear
run Data_multi.m

subplot(2,2,3)
axis([-1 2 -1 2])
grid on
subplot(2,2,4)
axis([-1 2 -1 2])
grid on

subplot(2,2,3)
hold on
title('Dados treino')
plot(X_A(:,1),X_A(:,2),'bs')
plot(X_B(:,1),X_B(:,2),'r+')
plot(X_C(:,1),X_C(:,2),'ko')

X_A = [X_A ones(size(X_A,1),1)];
X_B = [X_B ones(size(X_B,1),1)];
X_C = [X_C ones(size(X_C,1),1)];

subplot(2,2,4)
hold on
title('Dados Teste')
plot(X_teste(:,1),X_teste(:,2),'g*')

Layer1 = 3;
Layer2 = 5;
Layer3 = 3;


n_epochs=10000;  %numero de epocas
alpha = 0.9;    %Factor de aprendizagem

%Amostras de entrada
X = [X_A
    X_B
    X_C];

%Saida associada a cada vetor
T = [ones(size(X_A,1),1) zeros(size(X_A,1),1) zeros(size(X_A,1),1)
    zeros(size(X_B,1),1) ones(size(X_B,1),1) zeros(size(X_B,1),1)
    zeros(size(X_C,1),1) zeros(size(X_C,1),1) ones(size(X_C,1),1)];

%T = 2*T-1

%vector soma dos erros quadraticos
SSE=zeros(n_epochs,1);

N = size(X,1);  %numero de amostras

 
%Inicializaçao aleatoria dos pesos [-1,1]
W1 = 2*rand(Layer2-1,Layer1) - 1;
W2 = 2*rand(Layer3,Layer2) - 1;

%% propagaçao Feedforward

for epoch = 1:n_epochs
    sum_sq_error=0;
    for k = 1:N
        x = X(k,:)';
        t = T(k,:)'; %MUDAR AQUI
        
        %soma da camada de entrada
        g1 = W1*x;
        %Funçao de ativaçao sigmoidal
        y1 = ActFunc(g1);
        %y1 = tanh(g1);
        % Adiçao à saida da camada escondida y1
        % da entrada de bias com +1
        % Resulta em y1_b
        y1_b = [y1
                1];
            
        % Soma da camada de saida
        g2 = W2*y1_b;
        % Funçao de ativaçao sigmoidal
        y2 = ActFunc2(g2);
        
        %% retropropagaçao
        
        % Erro da camada de saida
        e = t - y2;
        
        %calculo do delta da camada de saida
        % sigmoide
        delta2 = y2.*(1-y2).*e;
        
        % atualizaçao da soma dos erros quadraticos
        sum_sq_error = sum_sq_error + sum(e.^2);
        
        %erro da camada escondida
        e1 = W2'*delta2;
        
        %erro sem o bias
        e1_b = e1(1:Layer2-1);  %MUDAR AQUI
        %e1_b = e1(1:2);
        
        %calculo do delta da camada de saida
        delta1 = y1.*(1-y1).*e1_b;
        
        % atualizaçao dos pesos da camada escondida
        dW2 = alpha*delta2*y1_b'; %com bias
        W2 = W2 + dW2;
        
        % Atualizaçao dos pesos da cama de entrada
        dW1 = alpha*delta1*x';
        W1 = W1 + dW1;
        
    end
    SSE(epoch) = sum_sq_error/N; 
end


V = [X_teste ones(size(X_teste,1),1)];
V = [V
     X];
N = size(V,1);
y_plot=zeros(N,Layer3);

for k = 1:N
    
    x = V(k,:)';
    g1 =W1*x;
    
    %sigmoide
    y1 = ActFunc(g1);
    
    % y1 mais uma entrada de bias
    y1_b = [y1
            1];
        
    g2 = W2*y1_b;
    
    % Saida prevista
    y_plot(k,:) = ActFunc2(g2');
end

subplot(2,2,[1,2], 'YScale', 'log')
hold on
It = 1:1:n_epochs;
plot(It,SSE,'r','LineWidth',2)
grid('on')
xlabel('Época')
ylabel('SSE')
title('Função de activação: Sigmóide')
hold off
[val,bigpos] = max(y_plot');

subplot(2,2,4)
hold on
for i = 1:N

    if bigpos(i)==1
        marker = 'bs';
    elseif bigpos(i)==2
        marker = 'r+';
    else
        marker = 'ko';
    end

    plot(V(i,1),V(i,2),marker)

end
hold off
subplot(2,2,[1,2])
hold off
subplot(2,2,3)
hold off
subplot(2,2,4)
hold off

%Escolha de funcao de ativacao
function [s]=ActFunc(x)
    s=sig(x);
end

function [s]=ActFunc2(x)
    s=sig(x);
end

%[0,1]
function [s]=sig(x)
    %Sigmoid
    s=1./(1+exp(-x));
end

function [s]=Gauss(x)
    %Gaussian
    s=exp(-x.^2);
end

%[-1,1]
function [s]=TangH(x)
    %HiperbolicTangent
    s = tanh(x);
end

function [s]=SoftSign(x)
    %SoftSign
    s = (x./(abs(x)+1));
end

%[0,inf]
function [s]=Softplus(x)
    %Softplus
    s = log(1+exp(x));
end

function [s]=ReLu(x)
    %ReLu
    s = max(0,x);
end

%[-inf,inf]
function [s]=BentIdentity(x)
    %Bent identity 
    s = ((sqrt(((x).^2)+1)-1)./2)+x;
end