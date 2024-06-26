#include <iostream>
#include <string>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>
#include <pcl/common/pca.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>

// Laden der Punktwolke
pcl::PointCloud<pcl::PointXYZRGB>::Ptr loadPointCloud(const std::string& filename) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    if (pcl::io::loadPLYFile<pcl::PointXYZRGB>(filename, *cloud) == -1) {
        PCL_ERROR("Couldn't read file %s \n", filename.c_str());
        return nullptr;
    }
    return cloud;
}

// Berechnung und Hinzufügung der Normalen zur Punktwolke
pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr calculateNormals(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud) {
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    pcl::copyPointCloud(*cloud, *cloud_with_normals);

    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>());
    ne.setSearchMethod(tree);

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    ne.setRadiusSearch(0.03);
    ne.compute(*normals);

    for (size_t i = 0; i < cloud_with_normals->points.size(); ++i) {
        cloud_with_normals->points[i].normal_x = normals->points[i].normal_x;
        cloud_with_normals->points[i].normal_y = normals->points[i].normal_y;
        cloud_with_normals->points[i].normal_z = normals->points[i].normal_z;
    }

    return cloud_with_normals;
}

// Zentrieren und Ausrichten der Punktwolke
void findAndAlignCentralAxis(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& cloud, float x_adjustment = 0.0f, float y_adjustment = 0.0f, float z_adjustment = 0.0f) {
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud, centroid);

    Eigen::Matrix4f translation = Eigen::Matrix4f::Identity();
    translation(0, 3) = -centroid[0] + x_adjustment;
    translation(1, 3) = -centroid[1] + y_adjustment;
    translation(2, 3) = -centroid[2] + z_adjustment;
    pcl::transformPointCloudWithNormals(*cloud, *cloud, translation);

    pcl::PCA<pcl::PointXYZRGBNormal> pca;
    pca.setInputCloud(cloud);
    Eigen::Matrix3f eigenVectors = pca.getEigenVectors();

    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block<3, 3>(0, 0) = eigenVectors.transpose();
    pcl::transformPointCloudWithNormals(*cloud, *cloud, transform);
}

// Berechnung des größten Radius
float calculateBiggestRadius(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& cloud) {
    float max_radius = 0.0;
    for (const auto& point : cloud->points) {
        float radius = std::sqrt(point.y * point.y + point.z * point.z);
        if (radius > max_radius) {
            max_radius = radius;
        }
    }
    return max_radius;
}

// Transformation der Punktwolke
void transformPointCloud(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& cloud, float biggest_radius) {
    for (auto& point : cloud->points) {
        float radius = std::sqrt(point.y * point.y + point.z * point.z);
        float theta = std::atan2(point.y, point.z); // Unwrapping around z-axis

        if (theta < 0) {
            theta += 2 * M_PI;
        }

        point.y = theta * biggest_radius;
        point.z = radius - biggest_radius;
    }
}

// Speichern der Punktwolke
bool savePointCloud(const std::string& filename, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr& cloud) {
    if (pcl::io::savePLYFile<pcl::PointXYZRGBNormal>(filename, *cloud) == -1) {
        PCL_ERROR("Couldn't write file %s \n", filename.c_str());
        return false;
    }
    return true;
}

// Hauptfunktion
int main(int argc, char** argv) {
    if (argc != 8) {
        std::cerr << "Usage: " << argv[0] << " <input_ply_file> <aligned_ply_file> <output_ply_file> <x_adjustment> <y_adjustment> <z_adjustment> <save_aligned>" << std::endl;
        return -1;
    }

    std::string input_filename = argv[1];
    std::string aligned_filename = argv[2];
    std::string output_filename = argv[3];
    float x_adjustment = std::stof(argv[4]);
    float y_adjustment = std::stof(argv[5]);
    float z_adjustment = std::stof(argv[6]);
    bool save_aligned = std::string(argv[7]) == "true";

    auto cloud = loadPointCloud(input_filename);
    if (!cloud) {
        return -1;
    }

    auto cloud_with_normals = calculateNormals(cloud);

    findAndAlignCentralAxis(cloud_with_normals, x_adjustment, y_adjustment, z_adjustment);
    
    if (save_aligned) {
        if (!savePointCloud(aligned_filename, cloud_with_normals)) {
            return -1;
        }
    }

    float biggest_radius = calculateBiggestRadius(cloud_with_normals);
    transformPointCloud(cloud_with_normals, biggest_radius);

    if (!savePointCloud(output_filename, cloud_with_normals)) {
        return -1;
    }

    std::cout << "Transformation complete.";
    if (save_aligned) {
        std::cout << " Aligned output saved to " << aligned_filename << ".";
    }
    std::cout << " Final output saved to " << output_filename << std::endl;
    return 0;
}
