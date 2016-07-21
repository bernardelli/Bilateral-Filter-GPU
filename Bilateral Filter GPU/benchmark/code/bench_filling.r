library(ggplot2) 
#lena
file_list <- Sys.glob("*lena_filling.txt")

file = file_list[3]

gpu_name = readLines(file, n=1) 

print(gpu_name)

data <- read.table(file, header = TRUE, sep = "\t", skip = 1)

#fprintf(output_file, "%s\nRepeat\tscale_xy\tscale_eps\tmethod\ttime\n", deviceProp.name);
data$method <- factor(data$method)
data$kernel_xy_size <- factor(data$method)
data$scale_eps <- factor(data$scale_eps)


data$method <- ordered(data$method,
levels = c(0,1),
labels = c("loop", "atomic")) 


ggplot(data, aes(x=scale_xy, y=time, color=method)) +
geom_point(alpha = 0.4) + 
stat_summary(fun.y = median, geom="line") + 
scale_colour_brewer(name="Method",palette = "Set1") +
ggtitle(paste("Filling on", gpu_name, "- Lena")) +# for the main title
xlab("Spatial downsample rate") + # for the x axis label
ylab("Runtime (ms)") + # for the y axis label
facet_grid(scale_eps ~ .) + ylim(0,23)

ggsave("out_filling1.png",width = 7.5, height = 5)

##random


file_list <- Sys.glob("*random_filling.txt")

file = file_list[3]

gpu_name = readLines(file, n=1) 

print(gpu_name)

data <- read.table(file, header = TRUE, sep = "\t", skip = 1)

#fprintf(output_file, "%s\nRepeat\tscale_xy\tscale_eps\tmethod\ttime\n", deviceProp.name);
data$method <- factor(data$method)
data$kernel_xy_size <- factor(data$method)
data$scale_eps <- factor(data$scale_eps)


data$method <- ordered(data$method,
levels = c(0,1),
labels = c("loop", "atomic")) 


ggplot(data, aes(x=scale_xy, y=time, color=method)) +
geom_point(alpha = 0.4) + 
stat_summary(fun.y = median, geom="line") + 
scale_colour_brewer(name="Method",palette = "Set1") +
ggtitle(paste("Filling on", gpu_name, "- Random")) +# for the main title
xlab("Spatial downsample rate") + # for the x axis label
ylab("Runtime (ms)") + # for the y axis label
facet_grid(scale_eps ~ .) +ylim(0,23)

ggsave("out_filling2.png",width = 7.5, height = 5)