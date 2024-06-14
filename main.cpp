#include <iostream>
#include <string>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>
#include <pcl/common/pca.h>
#include <pcl/common/transforms.h>
#include <cmath>

pcl::PointCloud<pcl::PointXYZRGB>::Ptr loadPointCloud(const std::string& filename) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    if (pcl::io::loadPLYFile<pcl::PointXYZRGB>(filename, *cloud) == -1) {
        PCL_ERROR("Couldn't read file %s \n", filename.c_str());
        return nullptr;
    }
    return cloud;
}

void findAndAlignCentralAxis(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, float x_adjustment = 0.0f, float y_adjustment = 0.0f, float z_adjustment = 0.0f) {
    // Calculate the centroid of the entire point cloud
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid);

    // Translate the point cloud to center it along the x-axis
    Eigen::Matrix4f translation = Eigen::Matrix4f::Identity();
    translation(0, 3) = -centroid[0] + x_adjustment;
    translation(1, 3) = -centroid[1] + y_adjustment;
    translation(2, 3) = -centroid[2] + z_adjustment;
    pcl::transformPointCloud(*cloud, *cloud, translation);

    // Perform PCA to align the principal component with the x-axis
    pcl::PCA<pcl::PointXYZRGB> pca;
    pca.setInputCloud(cloud);
    Eigen::Matrix3f eigenVectors = pca.getEigenVectors();

    // Align the principal axis with the x-axis
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block<3, 3>(0, 0) = eigenVectors.transpose();

    // Transform the point cloud to align it
    pcl::transformPointCloud(*cloud, *cloud, transform);
}

float calculateBiggestRadius(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud) {
    float max_radius = 0.0;
    for (const auto& point : cloud->points) {
        float radius = std::sqrt(point.y * point.y + point.z * point.z);
        if (radius > max_radius) {
            max_radius = radius;
        }
    }
    return max_radius;
}

void transformPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, float biggest_radius) {
    for (auto& point : cloud->points) {
        float radius = std::sqrt(point.y * point.y + point.z * point.z);
        float theta = std::atan2(point.y, point.z); // Unwrapping around z-axis

        // Adjust theta to start unwrapping from the plus z-axis
        if (theta < 0) {
            theta += 2 * M_PI;
        }

        // Map theta to a linear coordinate (unwrap) starting from the point on the plus z-axis that is equal to the biggest radius
        point.y = theta * biggest_radius;
        point.z = radius - biggest_radius;
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
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0] << " <input_ply_file> <aligned_ply_file> <output_ply_file> <x_adjustment> <y_adjustment> <z_adjustment>" << std::endl;
        return -1;
    }

    std::string input_filename = argv[1];
    std::string aligned_filename = argv[2];
    std::string output_filename = argv[3];
    float x_adjustment = std::stof(argv[4]);
    float y_adjustment = std::stof(argv[5]);
    float z_adjustment = std::stof(argv[6]);

    auto cloud = loadPointCloud(input_filename);
    if (!cloud) {
        return -1;
    }

    findAndAlignCentralAxis(cloud, x_adjustment, y_adjustment, z_adjustment);
    
    // Save the point cloud after aligning the central axis
    if (!savePointCloud(aligned_filename, cloud)) {
        return -1;
    }

    float biggest_radius = calculateBiggestRadius(cloud);
    transformPointCloud(cloud, biggest_radius);

    if (!savePointCloud(output_filename, cloud)) {
        return -1;
    }

    std::cout << "Transformation complete. Aligned output saved to " << aligned_filename << " and final output saved to " << output_filename << std::endl;
    return 0;
}
