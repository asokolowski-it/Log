#include <iostream>
#include <string>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>
#include <pcl/common/pca.h>
#include <pcl/common/transforms.h>

pcl::PointCloud<pcl::PointXYZRGB>::Ptr loadPointCloud(const std::string& filename) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    if (pcl::io::loadPLYFile<pcl::PointXYZRGB>(filename, *cloud) == -1) {
        PCL_ERROR("Couldn't read file %s \n", filename.c_str());
        return nullptr;
    }
    return cloud;
}

void findAndAlignCentralAxis(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud) {
    // Calculate the centroid of the point cloud
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid);

    // Perform Principal Component Analysis (PCA) to find the principal axes
    pcl::PCA<pcl::PointXYZRGB> pca;
    pca.setInputCloud(cloud);
    Eigen::Matrix3f eigenVectors = pca.getEigenVectors();

    // Align the principal axis with the x-axis
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block<3, 3>(0, 0) = eigenVectors;
    transform.block<3, 1>(0, 3) = -1.0f * (eigenVectors * centroid.head<3>());

    // Transform the point cloud to align it
    pcl::transformPointCloud(*cloud, *cloud, transform);
}

void transformPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud) {
    for (auto& point : cloud->points) {
        float radius = std::sqrt(point.y * point.y + point.z * point.z);
        float theta = std::atan2(point.z, point.y);
        // Map theta to a linear coordinate (unwrap)
        point.y = radius * theta;
        // Set z to zero as it is the unwrapped angle dimension, keep x as the length along the log
        point.z = 0.0;
    }
}

bool savePointCloud(const std::string& filename, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud) {
    if (pcl::io::savePLYFile<pcl::PointXYZRGB>(filename, *cloud) == -1) {
        PCL_ERROR("Couldn't write file %s \n", filename.c_str());
        return false;
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_ply_file> <output_ply_file>" << std::endl;
        return -1;
    }

    std::string input_filename = argv[1];
    std::string output_filename = argv[2];

    auto cloud = loadPointCloud(input_filename);
    if (!cloud) {
        return -1;
    }

    findAndAlignCentralAxis(cloud);
    transformPointCloud(cloud);

    if (!savePointCloud(output_filename, cloud)) {
        return -1;
    }

    std::cout << "Transformation complete. Output saved to " << output_filename << std::endl;
    return 0;
}
