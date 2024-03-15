% 读取 .mat 文件
trial = load('5kg_division3_new.mat').Trial;
data.time = trial.time';
data.torque = trial.Torque';
data.BIClon_Multipliers = trial.BIC_lon.Multipliers';
data.BIClon_Activation = trial.BIC_lon.Activation';
data.BIClon_MmtArm = trial.BIC_lon.MmtArm';
data.BICsht_Multipliers = trial.BIC_sht.Multipliers';
data.BICsht_Activation = trial.BIC_sht.Activation';
data.BICsht_MmtArm = trial.BIC_sht.MmtArm';
data.BRA_Multipliers = trial.BRA.Multipliers';
data.BRA_Activation = trial.BRA.Activation';
data.BRA_MmtArm = trial.BRA.MmtArm';
data.BRD_Multipliers = trial.BRD.Multipliers';
data.BRD_Activation = trial.BRD.Activation';
data.BRD_MmtArm = trial.BRD.MmtArm';
data.TRIlon_Multipliers = trial.TRI_lon.Multipliers';
data.TRIlon_Activation = trial.TRI_lon.Activation';
data.TRIlon_MmtArm = trial.TRI_lon.MmtArm';
data.TRIlat_Multipliers = trial.TRI_lat.Multipliers';
data.TRIlat_Activation = trial.TRI_lat.Activation';
data.TRIlat_MmtArm = trial.TRI_lat.MmtArm';
data.TRImed_Multipliers = trial.TRI_med.Multipliers';
data.TRImed_Activation = trial.TRI_med.Activation';
data.TRImed_MmtArm = trial.TRI_med.MmtArm';

data_table = struct2table(data);
var_names = fieldnames(data);

% 创建一个 Excel 文件
filename = '5kg_division3_new.xlsx';
sheet_name = 'Sheet1';

% 获取变量的最大尺寸
max_size = 0;
for i = 1:numel(var_names)
    var_data = data.(var_names{i});
    max_size = max(max_size, numel(var_data));
end

% 创建一个带有表头的单元格数组
cell_array = cell(max_size + 1, numel(var_names));

% 添加表头
for i = 1:numel(var_names)
    cell_array{1, i} = var_names{i};
end

% 将每个变量保存到单元格数组中
for i = 1:numel(var_names)
    var_data = data.(var_names{i});
    cell_array(2:numel(var_data) + 1, i) = num2cell(var_data);
end

% 将单元格数组保存为 .xlsx 文件
writecell(cell_array, filename, 'Sheet', sheet_name);

% % 遍历每个变量并保存为不同的工作表var_names = fieldnames(data);
% for i = 1:numel(var_names)
%     var_name = var_names{i};
%     var_data = data.(var_name);
% 
%     % 检查数据类型
%     if isnumeric(var_data) || islogical(var_data)
%         % 数值或逻辑数组：直接保存为一个工作表
%         xlswrite(filename, var_data, var_name);
% 
%     elseif iscell(var_data)
%         % 元胞数组：将每个元胞保存为一个工作表
%         for j = 1:numel(var_data)
%             cell_data = var_data{j};
%             sheet_name = sprintf('%s_cell_%d', var_name, j);
%             xlswrite(filename, cell_data, sheet_name);
%         end
% 
%     elseif ischar(var_data)
%         % 字符数组：保存为一个工作表
%         xlswrite(filename, {var_data}, var_name);
% 
%     elseif isstruct(var_data)
%         % 结构体：逐个字段保存为不同的工作表
%         field_names = fieldnames(var_data);
%         for j = 1:numel(field_names)
%             field_name = field_names{j};
%             field_data = var_data.(field_name);
%             sheet_name = sprintf('%s_%s', var_name, field_name);
%             xlswrite(filename, field_data, sheet_name);
%         end
% 
%     else
%         % 其他类型的数据：忽略或进行适当处理
%         disp(['Ignoring variable: ' var_name]);
%     end
% end

% var_names = fieldnames(data);
% for i = 1:numel(var_names)
%     var_name = var_names{i};
%     var_data = data.(var_name);
% 
%     % 检查是否为多维数组
%     if ismatrix(var_data)
%         % 将数据保存为一个工作表
%         xlswrite(filename, var_data, var_name);
%     else
%         % 遍历每个维度并保存为不同的工作表
%         for j = 1:size(var_data, ndims(var_data))
%             dim_name = sprintf('%s_dim_%d', var_name, j);
%             dim_data = squeeze(var_data(:, :, j));  % 提取当前维度的数据
% 
%             xlswrite(filename, dim_data, dim_name);
%         end
%     end
% end