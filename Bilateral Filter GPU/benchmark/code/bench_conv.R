library(ggplot2) 

file_list <- Sys.glob("*kom.txt")

file = file_list[1]

gpu_name = readLines(file, n=1) 

print(gpu_name)

data <- read.table(file, header = TRUE, sep = "\t", skip = 1)

data_subset <- subset( data, scale_eps == 11 & scale_xy == 11 & kernel_eps_size == 11 & kernel_xy_size > 1)

data_subset$Image.size <- factor(data_subset$Image.size)
#TRY MEDIAN INSTEAD OF MEAN!

#conv
ggplot(data_subset, aes(x=kernel_xy_size, y=convolution, color=Image.size)) +
geom_point(alpha = 0.4) + 
stat_summary(fun.y = median, geom="line") + 
scale_colour_brewer(name="Image size",palette = "Set3") +
ggtitle(paste("Convolution on", gpu_name)) +# for the main title
xlab("Spatial kernel length") + # for the x axis label
ylab("Runtime (ms)") + # for the y axis label


ggsave("out1_conv.png",width = 7.5, height = 5)
